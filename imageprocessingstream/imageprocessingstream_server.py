from concurrent import futures
import time
import logging

import cv2
import grpc
import numpy as np

import imageprocessing_pb2
import imageprocessing_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ImageProcessingStream(imageprocessing_pb2_grpc.ImageProcessingStreamServicer):

    def ToGrayscale(self, request_iterator, context):
        for request in request_iterator:
            src = np.frombuffer(request.data, dtype=np.uint8)
            src = src.reshape((request.height, request.width, request.channels))
            dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            yield imageprocessing_pb2.Image(height=request.height, width=request.width, channels=1, data=dst.tobytes())


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    imageprocessing_pb2_grpc.add_ImageProcessingStreamServicer_to_server(ImageProcessingStream(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
