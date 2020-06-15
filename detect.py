# run using
# python3 yolov5/detect.py --source /kaggle/input/global-wheat-detection/test/ --weights /kaggle/input/yolov5-v0/weights/best.pt --conf 0.1
# python3 detect.py --source=/Users/elimgta/Downloads/test/ --weights=/Users/elimgta/Downloads/best.pt --conf=0.1

import argparse

from utils.datasets import *
from utils.utils import *
import pandas as pd


def detect(source, weights, half, imgsz):
    """Returns map from a input name to a list of predicted boxes."""
    predictions = {}

    # Initialize
    device = torch_utils.select_device(opt.device)

    # Load model
    # google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, _ in dataset:
        predictions[path] = []
        try:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # to float
            if half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       fast=True, agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    # Save results
                    for *xyxy, conf, cls in det:
                        box = xyxy
                        predictions[path].append((conf.tolist(), int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])))
        except:
            pass

    print('Done. (%.3fs)' % (time.time() - t0))
    return predictions


def save_output(source, imgsz, out, predictions):
    if out:
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)
    results = []
    # for path, img, im0s, _ in dataset:
    for path, preds in predictions.items():
        # Locate tuple 
        pstr = []

        # save_path = ''
        # if out:
        #     save_path = str(Path(out) / Path(path).name)
        for pred in preds:
            if pred[0] >= 0.5:
                pstr.append('%0.2f %d %d %d %d' % (pred[0], pred[1], pred[2], pred[3], pred[4]))
            # if out:
            #     label = '%0.2f' % pred[0]
            #     plot_one_box((pred[1], pred[2], pred[1]+pred[3], pred[2]+pred[4]), im0s, label=label, color=(128, 128, 128), line_thickness=3)
        # # Add bbox to image
        # label = '%s %.2f' % (names[int(cls)], conf)
        # Save results (image with detections)
        # if out:
        #     cv2.imwrite(save_path, im0s)
        result = {
            'image_id': os.path.splitext(os.path.basename(path))[0],
            'PredictionString': ' '.join(pstr),
        }
        results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.head()
    test_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        predictions = detect(opt.source, opt.weights, opt.half, opt.img_size)
        # print(predictions)
        save_output(opt.source, opt.img_size, opt.output, predictions)
