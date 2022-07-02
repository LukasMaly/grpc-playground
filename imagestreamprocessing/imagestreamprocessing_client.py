from __future__ import print_function
import logging

import cv2
import grpc
import numpy as np

import imagestreamprocessing_pb2
import imagestreamprocessing_pb2_grpc


def generate_stream():
    src = cv2.imread('../data/peppers.tiff')
    height, width = src.shape[:2]
    channels = src.shape[2]
    for i in range(100):
        img = src.copy()
        img = cv2.putText(img, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        data = img.tobytes()
        yield imagestreamprocessing_pb2.Image(height=height, width=width, channels=channels, data=data)


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = imagestreamprocessing_pb2_grpc.ImageStreamProcessingStub(channel)
        responses = stub.ToGrayscale(generate_stream())
        for response in responses:
            dst = np.frombuffer(response.data, dtype=np.uint8)
            dst = dst.reshape((response.height, response.width))
            cv2.imshow('Output', dst)
            cv2.waitKey(42)

if __name__ == '__main__':
    logging.basicConfig()
    run()
