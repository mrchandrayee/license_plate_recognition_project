import argparse, time
from pathlib import Path
import cv2
from .pipeline import LicensePlatePipeline, load_config

def main():
    p=argparse.ArgumentParser(description="License Plate Recognition (ALPR)")
    p.add_argument("--config",default="configs/alpr.yaml")
    p.add_argument("--source",help="Override source in YAML: video path, image path, folder, or camera index")
    p.add_argument("--output",help="Override output directory")
    p.add_argument("--mode",choices=["full","detect"])
    p.add_argument("--no-show",action="store_true")
    p.add_argument("--save-video",action="store_true")
    args=p.parse_args()
    cfg=load_config(args.config); run=cfg["run"]
    if args.source: run["source"]=args.source
    if args.output: run["output"]=args.output
    if args.mode: run["mode"]=args.mode
    if args.no_show: run["show"]=False
    if args.save_video: run["save_video"]=True
    source=run.get("source","0")
    src=Path(str(source))
    output=Path(run.get("output","runs/alpr")); output.mkdir(parents=True,exist_ok=True)
    pipeline=LicensePlatePipeline(cfg)
    is_camera=str(source).isdigit() and not src.exists()
    if src.is_dir():
        paths=[f for f in sorted(src.rglob("*")) if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
        for path in paths:
            frame=cv2.imread(str(path))
            if frame is None: continue
            vis=pipeline.process_frame(frame)
            cv2.imwrite(str(output/f"{path.stem}_alpr.jpg"),vis)
        print(f"Processed {len(paths)} images into {output}")
        return
    if src.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
        frame=cv2.imread(str(src))
        if frame is None: raise RuntimeError(f"Cannot read image: {src}")
        cv2.imwrite(str(output/f"{src.stem}_alpr.jpg"),pipeline.process_frame(frame))
        return
    cap=cv2.VideoCapture(int(source) if is_camera else str(source))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open source: {source}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=None
    if run.get("save_video"):
        writer=cv2.VideoWriter(str(output/"annotated_alpr.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    count=0; start=time.time()
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            elapsed=max(time.time()-start,1e-6)
            vis=pipeline.process_frame(frame,(count+1)/elapsed)
            if writer: writer.write(vis)
            if run.get("show",True):
                cv2.imshow("License Plate Recognition",vis)
                if cv2.waitKey(1)&0xFF==ord("q"): break
            count+=1
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
    print(f"Processed {count} frames. Output: {output}")

if __name__=="__main__":
    main()
