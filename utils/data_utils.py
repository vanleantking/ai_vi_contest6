# import pickle
import pandas as pd

import re
DATA_FILE = '../data/train.crash'


def load_file():

	split_re = re.compile("train_\\d{2,}")
	all_reviews = list()
	with open(DATA_FILE, 'r', encoding="utf-8") as data_file:

		file_split = split_re.split(data_file.read())
		file_split = list(filter(lambda x: x != '', file_split))
		
		for label_data in file_split:
			d = {}
			data = label_data.split("\n")
			data = list(filter(lambda x: x != '', data))
			label = data[-1]
			text = ' '.join(data[:-1])
			review = process_data(text)
			d['text'] = review
			d['label'] = label
			# print(d)
			all_reviews.append(d)
	reviews_pd = pd.DataFrame.from_dict(all_reviews, orient='columns')
	return reviews_pd

def process_data(original):
	process = re.findall(r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s\d]+', original)
	process = list(map(lambda x: x.strip(), process))
	return process

print(load_file())