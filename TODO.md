# Dance Video Resizer

## Overview
Often, people make a portrait video of my dance competitions, which can not be uploaded as normal videos to YouTube. This project is to resize the video into a 16:9 format, which I can use to upload to YouTube.

## Functionalities

1. Resize the video to 16:9 format. Keep the aspect ratio of the original video.
2. Add dark enlarged (full width) video on the sides if necessary.
3. The goal of the video is to show the whole body of the dance couple in the front. On the other hand, the background is not important. So the video should be resized to make the dance couple as large as possible in the video. Resizing can happen per 3 seconds and can have a smooth transition between different sizes. 

## Implementation Notes

1. Use Python for implementation.
2. Use OpenCV to read the video and get the frames.
3. Use OpenCV to resize the frames and add the dark enlarged video on the sides if necessary.
4. Use OpenCV to write the resized frames into a new video file.
5. Use FFmpeg to handle the audio stream and merge it with the resized video if necessary.
6. Use open-source models for detecting dancers in the video, such as OpenPose or MediaPipe, to ensure that the dancers are properly framed in the resized video.
7. The input video file path and the output video file path should be provided as command line arguments.
8. The code should be modular and well-documented.
9. The code should handle edge cases, such as when the input video is already in 16:9 format or when the input video has a very small resolution.