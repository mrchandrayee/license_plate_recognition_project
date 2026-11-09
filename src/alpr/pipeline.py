from __future__ import annotations
import re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "object"
    track_id: Optional[int] = None

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1+x2)/2, (y1+y2)/2)

@dataclass
class PlateReading:
    text: str
    confidence: float
    plate_box: tuple[int, int, int, int]

@dataclass
class PlateCard:
    track_id: str
    text: str
    confidence: float
    last_seen: float = field(default_factory=time.time)

class OCR:
    def __init__(self, engine="easyocr", lang="en"):
        self.engine = engine
        if engine == "easyocr":
            import easyocr
            self.reader = easyocr.Reader([lang], gpu=False)
        elif engine == "tesseract":
            import pytesseract
            self.reader = pytesseract
        else:
            raise ValueError("ocr_engine must be easyocr or tesseract")

    @staticmethod
    def normalize(text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def read(self, image):
        if image is None or image.size == 0:
            return "", 0.0
        h, w = image.shape[:2]
        if max(h, w) < 900:
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        if self.engine == "easyocr":
            results = self.reader.readtext(gray)
            candidates = [(self.normalize(txt), float(conf)) for _, txt, conf in results]
        else:
            data = self.reader.image_to_data(gray, output_type=self.reader.Output.DICT,
                config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            candidates = [(self.normalize(txt), max(0, float(conf))/100)
                          for txt, conf in zip(data["text"], data["conf"]) if str(conf) != "-1"]
        candidates = [(txt, conf) for txt, conf in candidates if len(txt) >= 4]
        if not candidates:
            return "", 0.0
        # Prefer a long alphanumeric plate string with a strong OCR confidence.
        return max(candidates, key=lambda x: (len(x[0]), x[1]))

class LicensePlatePipeline:
    """Vehicle -> plate detector -> OCR pipeline with optional tracking and HUD."""
    def __init__(self, cfg):
        self.cfg = cfg
        run, det, alpr, trk = cfg["run"], cfg["detector"], cfg["alpr"], cfg.get("tracker", {})
        self.mode = run.get("mode", "full")
        if self.mode not in ("full", "detect"):
            raise ValueError("run.mode must be full or detect")
        self.scope = alpr.get("plate_scope", "vehicle-crop")
        if self.scope not in ("vehicle-crop", "full-frame"):
            raise ValueError("plate_scope must be vehicle-crop or full-frame")
        self.vehicle_model = YOLO(det["vehicle_weights"])
        self.plate_model = YOLO(det["plate_weights"])
        self.ocr = OCR(alpr.get("ocr_engine", "easyocr"), alpr.get("ocr_lang", "en"))
        self.cache_mode = alpr.get("ocr_cache", "once")
        self.cache = {}
        self.cards = {}
        self.frame_no = 0
        self.tracker_enabled = self.mode == "full"
        self.track_cfg = trk
        self.vehicle_names = set(det.get("vehicle_classes") or [])
        self.device = run.get("device")
        self.imgsz = run.get("imgsz", 960)
        self.conf = run.get("confidence", .25)
        self.iou = run.get("iou", .45)
        self.plate_conf = det.get("plate_confidence", .25)
        self.plate_imgsz = det.get("plate_imgsz", self.imgsz)
        self.padding = float(alpr.get("crop_padding", .08))
        self.zone = self._poly(alpr.get("detection_zone"))
        self.capture_line = alpr.get("capture_line")
        self.show_zone = bool(alpr.get("show_detection_zone", True))
        self.show_labels = bool(alpr.get("show_labels", True))
        self.hud = alpr.get("hud", {})
        self.prev_centers = {}

    @staticmethod
    def _poly(points):
        return None if not points else np.asarray(points, dtype=np.int32)

    def _predict(self, model, frame, conf, imgsz):
        result = model.predict(frame, conf=conf, iou=self.iou, imgsz=imgsz,
                               device=self.device, verbose=False)[0]
        out = []
        if result.boxes is None:
            return out
        names = result.names
        for b in result.boxes:
            xyxy = tuple(float(v) for v in b.xyxy[0].cpu().tolist())
            cls = int(b.cls[0].cpu())
            name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            out.append(Detection(xyxy, float(b.conf[0].cpu()), cls, str(name)))
        return out

    def _vehicles(self, frame):
        dets = self._predict(self.vehicle_model, frame, self.conf, self.imgsz)
        if self.vehicle_names:
            dets = [d for d in dets if d.class_name.lower() in self.vehicle_names]
        return dets

    def _in_zone(self, det):
        if self.zone is None:
            return True
        return cv2.pointPolygonTest(self.zone, det.center, False) >= 0

    def _plate_detections(self, frame, vehicles):
        results = []
        if self.scope == "full-frame":
            return [(d, frame, (0,0)) for d in self._predict(self.plate_model, frame, self.plate_conf, self.plate_imgsz)]
        h, w = frame.shape[:2]
        pad = self.padding
        for vehicle in vehicles:
            x1,y1,x2,y2 = vehicle.bbox
            dx,dy=(x2-x1)*pad,(y2-y1)*pad
            left,top=max(0,int(x1-dx)),max(0,int(y1-dy))
            right,bottom=min(w,int(x2+dx)),min(h,int(y2+dy))
            crop=frame[top:bottom,left:right]
            if crop.size == 0: continue
            for plate in self._predict(self.plate_model, crop, self.plate_conf, self.plate_imgsz):
                px1,py1,px2,py2=plate.bbox
                plate.bbox=(px1+left,py1+top,px2+left,py2+top)
                results.append((plate, frame, (0,0)))
        return results

    def _ocr_plate(self, frame, det):
        x1,y1,x2,y2=map(int,det.bbox)
        h,w=frame.shape[:2]
        px=max(3,int((x2-x1)*.08)); py=max(3,int((y2-y1)*.15))
        crop=frame[max(0,y1-py):min(h,y2+py),max(0,x1-px):min(w,x2+px)]
        text,conf=self.ocr.read(crop)
        return PlateReading(text,conf,(x1,y1,x2,y2)) if text else None

    def _track_key(self, det, index):
        # YOLO tracking can be plugged in; stable approximate IDs are assigned by
        # matching each detection to the nearest prior center.
        cx,cy=det.center
        best_id=None; best_dist=1e9
        for tid,(ox,oy,seen) in self.prev_centers.items():
            dist=((cx-ox)**2+(cy-oy)**2)**.5
            if dist < best_dist and dist < 90 and self.frame_no-seen < 20:
                best_id,best_dist=tid,dist
        if best_id is None:
            best_id=f"V{len(self.prev_centers)+1}"
        self.prev_centers[best_id]=(cx,cy,self.frame_no)
        return best_id

    def _draw_zone(self, frame):
        if self.zone is not None and self.show_zone:
            cv2.polylines(frame,[self.zone],True,(255,180,0),2)
        if self.capture_line:
            pts=np.asarray(self.capture_line,dtype=np.int32)
            cv2.line(frame,tuple(pts[0]),tuple(pts[1]),(255,128,0),2)
        return frame

    def _draw_hud(self, frame, fps):
        hud=self.hud; x=int(hud.get("panel_x",8)); y=int(hud.get("panel_y",8))
        width=min(int(hud.get("panel_width",320)),frame.shape[1]-x)
        cards=sorted(self.cards.values(),key=lambda c:c.last_seen,reverse=True)[:int(hud.get("max_cards",6))]
        height=int(hud.get("panel_height") or (48+len(cards)*30))
        overlay=frame.copy()
        cv2.rectangle(overlay,(x,y),(x+width,min(frame.shape[0],y+height)),(15,20,28),-1)
        cv2.addWeighted(overlay,.78,frame,.22,0,frame)
        title=hud.get("title","Plates")
        cv2.putText(frame,f"{title} | FPS {fps:.1f}",(x+9,y+22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,230,255),1,cv2.LINE_AA)
        for i,c in enumerate(cards):
            cv2.putText(frame,f"{c.track_id}: {c.text}  {c.confidence:.0%}",
                (x+9,y+49+i*30),cv2.FONT_HERSHEY_SIMPLEX,.46,(80,255,140),1,cv2.LINE_AA)

    def process_frame(self, frame, fps=0.0):
        self.frame_no += 1
        vehicles=self._vehicles(frame)
        vehicles=[v for v in vehicles if self._in_zone(v)]
        vis=frame.copy()
        vis=self._draw_zone(vis)
        for v in vehicles:
            x1,y1,x2,y2=map(int,v.bbox)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,200,255),2)
            if self.show_labels:
                cv2.putText(vis,f"{v.class_name} {v.confidence:.0%}",(x1,max(18,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),1,cv2.LINE_AA)
        plates=self._plate_detections(frame,vehicles)
        for index,(plate,_,_) in enumerate(plates):
            x1,y1,x2,y2=map(int,plate.bbox)
            track_id=self._track_key(plate,index)
            key=track_id if self.cache_mode=="once" else f"{track_id}:{self.frame_no}"
            reading=self.cache.get(key)
            if reading is None and self.mode=="full":
                reading=self._ocr_plate(frame,plate)
                if reading:
                    self.cache[key]=reading
                    self.cards[track_id]=PlateCard(track_id,reading.text,reading.confidence)
            color=(0,255,160) if reading else (255,128,0)
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            label=reading.text if reading else "Plate"
            cv2.rectangle(vis,(x1,max(0,y1-24)),(min(vis.shape[1]-1,x1+max(70,len(label)*11)),y1),color,-1)
            cv2.putText(vis,label,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
        self._draw_hud(vis,fps)
        return vis

def load_config(path):
    import yaml
    with open(path,encoding="utf-8") as f:
        return yaml.safe_load(f)
