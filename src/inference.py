import os
import cv2
import sys
import glob
import argparse

from preprocess_for_inference import preprocess
from predict_for_inference import predict


def parse_args():
    parser = argparse.ArgumentParser(
            description='config for inference',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='checkpoint.pth',
                        help='model path')
    parser.add_argument('--device_id', type=str, default='cuda:0',
                        help='cpu or gpu device')
    parser.add_argument("--video_path", help="Video path or dir to evaluate on", type=str, default="./test_videos")
    parser.add_argument("--save_root", help="Directory to save img data", type=str, default="./output")
    parser.add_argument("--output_file_path", help="Path to save output", type=str, default="./result.txt")
    return parser.parse_args()

def main():
    args = parse_args()

    if os.path.isdir(args.video_path):
        video_paths = []
        for e in ['*.mp4', '*.avi']:
            video_paths.extend(glob.glob(os.path.join(args.video_path, e)))
    else:
        if os.path.isfile(args.video_path):
            video_paths = [args.video_path]
        else:
            print("File not found.")
            sys.exit('exit')
    if len(video_paths) == 0:
        print("No videos in the directory.")
        sys.exit('exit')

    with open(args.output_file_path, 'w') as f:
        for video_path in video_paths:
            preprocess(
                video_path=video_path,
                save_root=args.save_root,
            )

            probs = predict(
                data_root=args.save_root,
                model_path = args.model_path,
                device_id=args.device_id
            )

            video_name = video_path.split('/')[-1].split('.')[0]

            frame_dir = os.path.join(args.save_root, video_name, 'frames')
            temp_video_path = os.path.join(args.save_root, video_name, 'temp.mp4')
            img_names = sorted(os.listdir(frame_dir))
            w, h = cv2.imread(os.path.join(frame_dir, img_names[0])).shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vOut = cv2.VideoWriter(temp_video_path, fourcc, 25, (h, w))

            for i, img_name in enumerate(img_names):
                if i//5 >= len(probs[video_name]):
                    break
                img = cv2.imread(os.path.join(frame_dir, img_name))
                cv2.putText(
                    img,
                    f'fake prob: {probs[video_name][i//5]*100:.1f}', 
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA)
                vOut.write(img)
            
            vOut.release()
            audio_path = os.path.join(args.save_root, video_name, 'audio.wav')
            output_video_path = os.path.join(args.save_root, video_name, 'result.mp4')
            os.system(f'ffmpeg -i {temp_video_path} -i {audio_path} -c:v copy -c:a aac {output_video_path}')
            os.system(f'rm {temp_video_path}')


if __name__ == "__main__":
    main()