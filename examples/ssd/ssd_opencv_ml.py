import numpy as np
import cv2
import argparse

inWidth = 300
inHeight = 450
confThreshold = 0.5
mean = (104, 117, 123)

prefix = 'models/VGGNet/VOC0712/SSD_ml_300x300/'
prototxt = prefix + 'deploy.prototxt'
caffemodel = prefix + 'VGG_ml_SSD_ml_300x300_iter_10000.caffemodel'

if __name__ == '__main__':
    #Get image path
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default='examples/images/IMG0502.jpg')
    args = parser.parse_args()
    img = cv2.imread(args.image_file)
    cols = img.shape[1]
    rows = img.shape[0]

    #Init model
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    net.setInput(cv2.dnn.blobFromImage(img, 1.0, (inWidth, inHeight), mean, False, False))

    detections = net.forward("detection_out")
    perf_stats = net.getPerfProfile()

    print('Inference time, ms: %.2f' % (perf_stats[0] / cv2.getTickFrequency() * 1000))

    #Show detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confThreshold:
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            cv2.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
            print("Detection {}: {} {} {} {}".format(i, xLeftBottom, yLeftBottom, xRightTop, yRightTop))

            label = "gear: %.4f" % confidence
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(img, (xLeftBottom, yLeftBottom - labelSize[1]),
                                (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(img, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("detections", img)
    while True:
        k = cv2.waitKey(10) 
        if k == 27: #Escape
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty('detections',cv2.WND_PROP_VISIBLE) < 1:        
            break        

    


    
