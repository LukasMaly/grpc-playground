from __future__ import print_function
import logging

import cv2
import grpc
import numpy as np

import imageprocessing_pb2
import imageprocessing_pb2_grpc


def run():
    src = cv2.imread('../data/peppers.tiff')
    height, width = src.shape[:2]
    channels = src.shape[2]
    data = src.tobytes()
    cv2.imshow('Input', src)
    cv2.waitKey(1)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = imageprocessing_pb2_grpc.ImageProcessingStub(channel)
        img = imageprocessing_pb2.Image(height=height, width=width, channels=channels, data=data)
        response = stub.ToGrayscale(img)

    dst = np.frombuffer(response.data, dtype=np.uint8)
    dst = dst.reshape((response.height, response.width))

    cv2.imshow('Output', dst)
    cv2.waitKey()


if __name__ == '__main__':
    logging.basicConfig()
    run()
