
import os
from finders_tools.finders.tools.detection_utils import show_confusion
from finders_tools.finders.tools.url_utils import fetch_url_to_memory, load_json_url, url_utils

path = '/hdd/hanoch/results'
REF_QUALITY = 1
url = url_utils.parse_url(path + '/confusion_' + str(REF_QUALITY) + '_.json')
js = load_json_url(url)
show_confusion(js['confusion'], js['groundtruth_classes'], js['detection_classes'],'conf1')
os.environ['DISPLAY'] = str('localhost:10.0')