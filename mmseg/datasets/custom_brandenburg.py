import os.path as osp
import random
from functools import reduce
import os
import numpy as np
from terminaltables import AsciiTable
from torch.utils.data import Dataset
import scipy.io as sio
import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class BrandenburgDataset(Dataset):
    # todo
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.
    
    MTS12 DATASET: http://gpcv.whu.edu.cn/data/dataset12/dataset12.html

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = (
    # "Background",
    "Maize",
    "Wheat",
    "Grassland",
    "Peanut",
    "Potato",
    "Residue",
    "Fallow",
    "Eapeseed",
    "Vegetable",
    "Legume",
    "Herb",
    "Orchard",
    "Flower",
    "Sugar beet",
    "Other",)
    PALETTE = [
    # [255, 255, 255],
    [243, 173, 61],
    [209, 124, 248],
    [170, 226, 71],
    [235, 50, 35],
    [196, 140, 108],
    [198, 231, 253],
    [106, 43, 14],
    [42, 114, 246],
    [168, 167, 53],
    [204, 204, 204],
    [255, 253, 84],
    [209, 177, 161],
    [141, 221, 251],
    [188, 160, 211],
    [245, 193, 230],]


    def __init__(self,
                 pipeline,
                 img_dir,
                 split:str,
                 img_suffix='.mat',
                 ann_dir=None,
                 seg_map_suffix='.mat',
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.split = split
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not osp.isabs(self.split):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.ann_dir)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, ann_dir):
        """Load annotation from directory. Note that we do not use split for mts12 dataset!

        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        with open(self.split, 'r') as file:
            idx_list = [line.strip() for line in file.readlines()]
        idx_list = np.array(idx_list).astype(np.int32)
        imgs = np.array(sorted(os.listdir(img_dir)))
        anns = np.array(sorted(os.listdir(ann_dir)))
        imgs = imgs[idx_list]
        anns = anns[idx_list]
        for i, img in enumerate(imgs):
            img_info = dict(filename=img)
            # if ann_dir is not None:
            #     if 's1' in ann_dir:
            #         seg_map = img.replace('VHVV', 'label_s1')
            #     elif 's2' in ann_dir:
            #         seg_map = img.replace('VHVV', 'label_s2')
            seg_map=anns[i]
            assert seg_map[-8:] == img[-8:]
            img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    # TODO val or test
    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            # gt_seg_map = mmcv.imread(
            #     seg_map, flag='unchanged', backend='pillow')
            gt_seg_map = sio.loadmat(seg_map)['label']
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            # H,W -> D,H,W (D=12)
            # gt_seg_map = np.stack([gt_seg_map] * 12, axis=0)
            gt_seg_maps.append(gt_seg_map) # todo

        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            # !!!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!
            # !WE DO NOT USE self.label_map dict.!
            # !BECAUSE OUR SEG MAP IS RIGHT! DO NOT NEED MAP!!!!
            # self.label_map = {}
            # for i, c in enumerate(self.CLASSES):
            #     if c not in class_names:
            #         self.label_map[i] = -1
            #     else:
            #         self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mF1score']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        # metrics     = [    ~  None  ~         |~mIoU~|~f1score~| mDice ]
        # ret_metrics = [all_acc, acc, precision, iou,   f1score,   dice ]
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            ignore_index=self.ignore_index,
            metrics=metric)
        class_table_data = [['Class'] + ['Acc'] + ['Precision'] + [m[1:] for m in metric]]
        # if 'f1score' in metric:
        #     class_table_data += ['f1score']
        # if 'mDice' in metric:
        #     class_table_data += ['dice']

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes): # todo
            class_data = [class_names[i]] + [ret_metrics_round[j][i] for j in range(1, len(ret_metrics_round))]
            #     + [ret_metrics_round[3][i]]
            # if 'mF1score' in metric:
            #     class_data += [ret_metrics_round[4][i]]
            # if 'mDice' in metric:
            #     class_data += [ret_metrics_round[5][i]]
            class_table_data.append(class_data)

        summary_table_data = [['Scope'] + ['aAcc'] + 
                              ['m' + head
                               for head in class_table_data[0][1:]]]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_data = ['global'] + [mean_value for mean_value in ret_metrics_mean]
        summary_table_data.append(summary_data)
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
            [i]] = summary_table_data[1][i] / 100.0
        return eval_results
