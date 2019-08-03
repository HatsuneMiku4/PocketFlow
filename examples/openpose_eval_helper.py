import json
from pathlib import Path

import cv2
import numpy as np
import slidingwindow as sw
import tensorflow as tf
from pycocotools import COCO
from pycocotools.cocoeval import COCOeval
from tf_pose import common
from tf_pose.estimator import PoseEstimator
from tf_pose.tensblur.smoother import Smoother
from tqdm.auto import tqdm

FLAGS = tf.app.flags.FLAGS


class OpenPoseEstimator:
    UPSAMPLE_RATIO = 4.0
    TARGET_SIZE = (432, 368)
    UPSAMPLE_SIZE = int(TARGET_SIZE[1] / 8 * UPSAMPLE_RATIO), int(TARGET_SIZE[0] / 8 * UPSAMPLE_RATIO)

    def __init__(self, graph_fn):
        self.forward_fn = graph_fn
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensor_output = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 57))
            self.tensor_heatMat = self.tensor_output[:, :, :, :19]
            self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
            self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
            self.tensor_heatMat_up = tf.image.resize_area(
                self.tensor_output[:, :, :, :19], self.upsample_size, align_corners=False, name='upsample_heatmat')
            self.tensor_pafMat_up = tf.image.resize_area(
                self.tensor_output[:, :, :, 19:], self.upsample_size, align_corners=False, name='upsample_pafmat')
            gaussian_heatMat = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0).get_output()
            max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, (3, 3), 'MAX', 'SAME')
            self.tensor_peaks = tf.where(
                tf.equal(gaussian_heatMat, max_pooled_in_tensor),
                gaussian_heatMat, tf.zeros_like(gaussian_heatMat))

            self.persistent_sess = tf.Session(graph=self.graph)
            self.persistent_sess.run(tf.variables_initializer([
                v for v in tf.global_variables() if v.name.split(':')[0] in [
                    x.decode('utf-8') for x in self.persistent_sess.run(tf.report_uninitialized_variables())
                ]]))

        self.heatMat = self.pafMat = None

    def inference(self, npimg):

        def add_batch_dim(np_im): return np_im[None] if np_im.ndim == 3 else np_im

        output = self.forward_fn(add_batch_dim(self._get_scaled_img(npimg, None)[0][0]))

        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_output: output,
                self.upsample_size: self.UPSAMPLE_SIZE
            })

        return PoseEstimator.estimate_paf(peaks[0], heatMat_up[0], pafMat_up[0])

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.TARGET_SIZE[0] / float(h), self.TARGET_SIZE[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.TARGET_SIZE[1], self.TARGET_SIZE[0]):
                # resize
                npimg = cv2.resize(npimg, self.TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.TARGET_SIZE[1] or npimg.shape[1] < self.TARGET_SIZE[0]:
                newimg = np.zeros(
                    (max(self.TARGET_SIZE[1], npimg.shape[0]), max(self.TARGET_SIZE[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.TARGET_SIZE[0], self.TARGET_SIZE[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.TARGET_SIZE[1] or npimg.shape[1] < self.TARGET_SIZE[0]:
                newimg = np.zeros(
                    (max(self.TARGET_SIZE[1], npimg.shape[0]), max(self.TARGET_SIZE[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.TARGET_SIZE[0], self.TARGET_SIZE[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.TARGET_SIZE[0] / float(npimg.shape[1])
            ratio_h = self.TARGET_SIZE[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.TARGET_SIZE
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.TARGET_SIZE[1], self.TARGET_SIZE[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped


def calculate_map(graph_fn, subset=None):
    def write_coco_json(human, image_w, image_h):
        def round_int(val):
            return int(round(val))

        keypoints = []
        for coco_id in [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]:
            if coco_id not in human.body_parts.keys():
                keypoints.extend([0, 0, 0])
                continue
            body_part = human.body_parts[coco_id]
            keypoints += [round_int(body_part.x * image_w), round_int(body_part.y * image_h), 2]
        return keypoints

    cocoGt = COCO(FLAGS.coco_json_path)
    human_keys = cocoGt.getImgIds(catIds=cocoGt.getCatIds(catNms=['person']))
    subset = subset or len(human_keys)
    print(f'valid_set.size={subset}/{len(human_keys)}')
    human_keys = human_keys[:subset]

    image_dir = Path(FLAGS.image_dir) / 'val2017'
    if not image_dir.is_dir(): image_dir = Path(FLAGS.image_dir) / 'val'

    estimator = OpenPoseEstimator(graph_fn)
    results = []

    for k in tqdm(human_keys):
        img_meta = cocoGt.loadImgs(k)[0]
        img_name = str(image_dir / img_meta['file_name'])
        humans = estimator.inference(common.read_imgfile(img_name, None, None))
        results += [{
            'image_id': img_meta['id'],
            'category_id': 1,
            'keypoints': write_coco_json(human, img_meta['width'], img_meta['height']),
            'score': human.score,
        } for human in humans]

    with open('result.json', 'w') as fp:
        json.dump(results, fp)
    cocoDt = cocoGt.loadRes('result.json')
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = human_keys
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(''.join(["%11.4f |" % x for x in cocoEval.stats]))
