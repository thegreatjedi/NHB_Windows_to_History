import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.builders.model_builder import build
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.utils.label_map_util import \
    convert_label_map_to_categories, create_category_index, \
    get_max_label_map_index, load_labelmap
from six import BytesIO
from tensorflow import convert_to_tensor, function, io, reshape, train


def load_image(path: str) -> np.ndarray:
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    im_width, im_height = image.size
    
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
    """Get a tf.function for detection."""
    
    @function(experimental_relax_shapes=True)
    def detect_fn(image):
        """Detect objects in image."""
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)
        
        return detections, prediction_dict, reshape(shapes, [-1])
    
    return detect_fn


def init_detector():
    config_path = './obj_detect/efficientdet_d4_coco17_tpu-32.config'
    model_dir = './obj_detect/checkpoint/'
    
    # Build model
    configs = get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    detection_model = build(model_config=model_config, is_training=False)
    
    # Restore checkpoint
    ckpt = train.Checkpoint(model=detection_model)
    ckpt.restore(f'{model_dir}ckpt-0')
    
    detect_fn = get_model_detection_function(detection_model)
    
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = load_labelmap(label_map_path)
    categories = convert_label_map_to_categories(
        label_map, max_num_classes=get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = create_category_index(categories)
    
    return detect_fn, category_index


def detect_labels(img_path: str, detect_fn, category_index):
    img_np = load_image(img_path)
    input_tensor = convert_to_tensor(
        np.expand_dims(img_np, 0), dtype=tf.float32)
    detections, _, _ = detect_fn(input_tensor)
    
    classes = (detections['detection_classes'][0].numpy() + 1).astype(int)
    scores = detections['detection_scores'][0].numpy()
    labels = {category_index[classes[i]]['name']
              for i in range(scores.shape[0]) if scores[i] > 0.3}
    
    return labels
