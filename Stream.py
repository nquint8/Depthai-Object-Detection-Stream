
#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
from depthai_sdk import NNModelDescription
from depthai_sdk.managers import PipelineManager

# Create pipeline
with dai.Pipeline() as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build()
    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(cameraNode, dai.NNModelDescription("yolov11-small"))
    labelMap = detectionNetwork.getClasses()

    qRgb = detectionNetwork.passthrough.createOutputQueue()
    qDet = detectionNetwork.out.createOutputQueue()

    pipeline.start()

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections: # for each detection in the list of detections get the bounding box and label and confidence
            bbox = frameNorm(
                frame,
                (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
            cv2.putText(
                frame,
                labelMap[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"{int(detection.confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    while pipeline.isRunning(): #while pipeline is running get frames and detections from queues and display them
        inRgb: dai.ImgFrame = qRgb.get() # ge rb frame from the queue
        inDet: dai.ImgDetections = qDet.get() #get the detections from the queue
        if inRgb is not None: #if rgb is not null ,update the frame and put fps text on it
            frame = inRgb.getCvFrame()
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                color2,
            )

        if inDet is not None: #if detections are not null uodate the detection list and countero
            detections = inDet.detections
            counter += 1

        if frame is not None: #if frame is not null ,display it
            displayFrame("rgb", frame)
            print("FPS: {:.2f}".format(counter / (time.monotonic() - startTime)))
        if cv2.waitKey(1) == ord("q"): #quits program
            pipeline.stop()
            break
