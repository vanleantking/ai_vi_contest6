# import pickle
import pandas as pd

import re
from underthesea import word_tokenize

TRAIN_FILE = '../data/train.crash'
TEST_FILE = '../data/test.crash'

process_train = '../data/train.csv'
process_test = '../data/test.csv'
ABB_FILE = '../data/abb.txt'

replace_list = {"👹": "không tốt", "👻": "tốt", "💃": "tốt",'🤙': ' tốt ', '👍': ' tốt ',
"💄": "tốt", "💎": "tốt", "💩": "tốt","😕": "không tốt", "😱": "không tốt", "😸": "tốt",
"😾": "không tốt", "🚫": "không tốt",  "🤬": "không tốt","🧚": "tốt", "🧡": "tốt",'🐶':' tốt ',
'👎': ' không tốt ', '😣': ' không tốt ','✨': ' tốt ', '❣': ' tốt ','☀': ' tốt ', '♥': ' tốt ',
'🤩': ' tốt ', 'like': ' tốt ', '💌': ' tốt ', '🤣': ' tốt ', '🖤': ' tốt ', '🤤': ' tốt ',
':(': ' không tốt ', '😢': ' không tốt ', '❤': ' tốt ', '😍': ' tốt ', '😘': ' tốt ',
'😪': ' không tốt ', '😊': ' tốt ', '?': ' ? ', '😁': ' tốt ', '💖': ' tốt ', '😟': ' không tốt ',
'😭': ' không tốt ', '💯': ' tốt ', '💗': ' tốt ', '♡': ' tốt ', '💜': ' tốt ', '🤗': ' tốt ',
'^^': ' tốt ', '😨': ' không tốt ', '☺': ' tốt ', '💋': ' tốt ', '👌': ' tốt ', '😖': ' không tốt ',
'😀': ' tốt ', ':((': ' không tốt ', '😡': ' không tốt ', '😠': ' không tốt ', '😒': ' không tốt ',
'🙂': ' tốt ', '😏': ' không tốt ', '😝': ' tốt ', '😄': ' tốt ','😙': ' tốt ',
'😤': ' không tốt ', '😎': ' tốt ', '😆': ' tốt ', '💚': ' tốt ', '✌': ' tốt ', '💕': ' tốt ',
'😞': ' không tốt ', '😓': ' không tốt ', '️🆗️': ' tốt ', '😉': ' tốt ', '😂': ' tốt ',
':v': '  tốt ', '=))': '  tốt ', '😋': ' tốt ', '💓': ' tốt ', '😐': ' không tốt ', ':3': ' tốt ',
'😫': ' không tốt ', '😥': ' không tốt ', '😃': ' tốt ', '😬': ' 😬 ', '😌': ' 😌 ',
'💛': ' tốt ', '🤝': ' tốt ', '🎈': ' tốt ', '😗': ' tốt ', '🤔': ' không tốt ',
'😑': ' không tốt ', '🔥': ' không tốt ', '🙏': ' không tốt ', '🆗': ' tốt ', '😻': ' tốt ',
'💙': ' tốt ', '💟': ' tốt ', '😚': ' tốt ', '❌': ' không tốt ', '👏': ' tốt ', ';)': ' tốt ',
'<3': ' tốt ', '🌝': ' tốt ',  '🌷': ' tốt ', '🌸': ' tốt ', '🌺': ' tốt ', '🌼': ' tốt ',
'🍓': ' tốt ', '🐅': ' tốt ', '🐾': ' tốt ', '👉': ' tốt ', '💐': ' tốt ', '💞': ' tốt ',
'💥': ' tốt ', '💪': ' tốt ', '💰': ' tốt ',  '😇': ' tốt ', '😛': ' tốt ', '😜': ' tốt ',
'🙃': ' tốt ', '🤑': ' tốt ', '🤪': ' tốt ','☹': ' không tốt ',  '💀': ' không tốt ',
'😔': ' không tốt ', '😧': ' không tốt ', '😩': ' không tốt ', '😰': ' không tốt ',
'😳': ' không tốt ', '😵': ' không tốt ', '😶': ' không tốt ', '🙁': ' không tốt ',
'6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ',
'🎉': 'tốt', '5sao': ' 5star ', 'starstarstarstarstar': ' 5star ',
'1 sao': ' 1star ', '1sao': ' 1star ', '2 sao':' 1star ','2sao':' 1star ',
'2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star '}


def abb_file():
	abb_words = {}
	with open(ABB_FILE, 'r', encoding="utf-8") as data_file:
		file_split = data_file.read().split("\n")
		for line in file_split:
			data_line = line.split(":")
			values = data_line[1].strip()
			keys = data_line[0].split(",")
			for key in keys:
				abb_words[key.strip()] = values
	return abb_words

def test_file(dict_abbs, files, save_path):

	for file_type, file_path in files.items():
		if file_type == "test":
			split_re = re.compile("test_\\d{2,}")
		all_reviews = list()
		with open(file_path, 'r', encoding="utf-8") as data_file:
			data_test = data_file.read()
			labels = split_re.findall(data_test)

			file_split = split_re.split(data_test)
			file_split = list(filter(lambda x: x != '', file_split))
			# print(file_split)
			idx = 0
			for label_data in file_split:
				d = {}
				data = label_data.split("\n")
				data = list(filter(lambda x: x != '', data))
				text = ' '.join(data)
				# if text != "" :
				review = process_data(text, dict_abbs)
				d['text'] = review
				d['label'] = labels[idx]
				all_reviews.append(d)
				idx += 1
			reviews_pd = pd.DataFrame.from_dict(all_reviews, orient='columns')
			reviews_pd.to_csv(save_path[file_type], sep=',', encoding='utf-8',
				header=True, columns=['text', 'label'], index=False)

def load_file(dict_abbs, files, save_path):

	for file_type, file_path in files.items():
		if file_type == "test":
			split_re = re.compile("test_\\d{2,}")
		else:
			split_re = re.compile("train_\\d{2,}")
		all_reviews = list()
		with open(file_path, 'r', encoding="utf-8") as data_file:

			file_split = split_re.split(data_file.read())
			file_split = list(filter(lambda x: x != '', file_split))
			
			for label_data in file_split:
				d = {}
				data = label_data.split("\n")
				data = list(filter(lambda x: x != '', data))
				# print(label_data, data)
				if file_type == "train":
					label = data[-1]
					text = ' '.join(data[:-1])
				else:
					text = ' '.join(data)
				if text != "" :
					review = process_data(text, dict_abbs)
					d['text'] = review
					if file_type == "train":
						d['label'] = label
					all_reviews.append(d)
			reviews_pd = pd.DataFrame.from_dict(all_reviews, orient='columns')
			reviews_pd.to_csv(save_path[file_type], sep=',', encoding='utf-8',
				header=True, columns=['text', 'label'], index=False)

def process_data(original, dict_abbs):

	# replace i-con and star by text
	for k, v in replace_list.items():
		original = original.replace(k, v)
	process = UniStd(original)
	# process = re.findall('[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s\d\\.,!?\\-/]+', process)
	process = re.sub('[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ0-9 ]+', ' ', process)
	# process = re.findall(r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s\d]+', process)

	text = process.lower()
	for key, value in dict_abbs.items():
		# find word with spaces in text
		# if (' ' + key + ' ') in (' ' + text + ' '):
		# 	text = text.replace(' ' + key + ' ', ' ' + value + ' ')
		match_string = r'\b' + key + r'\b'
		regex = re.compile(match_string, re.S)
		text = regex.sub(lambda m: m.group().replace(key,value,1), text)
	text = text.replace('\xa0', '') # error in encode
	text = re.sub('[",]', '', text)
	text = re.sub(r'([a-z])\1+', r'',text) # remove duplicate charater in word: ơiiiiiiiiiii
	text = ' '.join(text.split())
	text = text.strip()
	text = word_tokenize(text, format="text")

	return text

def UniStd_L(str):
	return str.\
		replace(u'à',u'à').replace(u'ã',u'ã').replace(u'ả',u'ả').replace(u'á',u'á').replace(u'ạ',u'ạ').\
		replace(u'ằ',u'ằ').replace(u'ẵ',u'ẵ').replace(u'ẳ',u'ẳ').replace(u'ắ',u'ắ').replace(u'ặ',u'ặ').\
		replace(u'ầ',u'ầ').replace(u'ẫ',u'ẫ').replace(u'ẩ',u'ẩ').replace(u'ấ',u'ấ').replace(u'ậ',u'ậ').\
		replace(u'ỳ',u'ỳ').replace(u'ỹ',u'ỹ').replace(u'ỷ',u'ỷ').replace(u'ý',u'ý').replace(u'ỵ',u'ỵ').\
		replace(u'ì',u'ì').replace(u'ĩ',u'ĩ').replace(u'ỉ',u'ỉ').replace(u'í',u'í').replace(u'ị',u'ị').\
		replace(u'ù',u'ù').replace(u'ũ',u'ũ').replace(u'ủ',u'ủ').replace(u'ú',u'ú').replace(u'ụ',u'ụ').\
		replace(u'ừ',u'ừ').replace(u'ữ',u'ữ').replace(u'ử',u'ử').replace(u'ứ',u'ứ').replace(u'ự',u'ự').\
		replace(u'è',u'è').replace(u'ẽ',u'ẽ').replace(u'ẻ',u'ẻ').replace(u'é',u'é').replace(u'ẹ',u'ẹ').\
		replace(u'ề',u'ề').replace(u'ễ',u'ễ').replace(u'ể',u'ể').replace(u'ế',u'ế').replace(u'ệ',u'ệ').\
		replace(u'ò',u'ò').replace(u'õ',u'õ').replace(u'ỏ',u'ỏ').replace(u'ó',u'ó').replace(u'ọ',u'ọ').\
		replace(u'ờ',u'ờ').replace(u'ỡ',u'ỡ').replace(u'ở',u'ở').replace(u'ớ',u'ớ').replace(u'ợ',u'ợ').\
		replace(u'ồ',u'ồ').replace(u'ỗ',u'ỗ').replace(u'ổ',u'ổ').replace(u'ố',u'ố').replace(u'ộ',u'ộ').\
		replace(u'òa',u'oà').replace(u'õa',u'oã').replace(u'ỏa',u'oả').replace(u'óa',u'oá').replace(u'ọa',u'oạ').\
		replace(u'òe',u'oè').replace(u'õe',u'oẽ').replace(u'ỏe',u'oẻ').replace(u'óe',u'oé').replace(u'ọe',u'oẹ').\
		replace(u'ùy',u'uỳ').replace(u'ũy',u'uỹ').replace(u'ủy',u'uỷ').replace(u'úy',u'uý').replace(u'ụy',u'uỵ').\
		replace(u'aó',u'áo')

def UniStd_H(str):
	return str.\
		replace(u'À',u'À').replace(u'Ã',u'Ã').replace(u'Ả',u'Ả').replace(u'Á',u'Á').replace(u'Ạ',u'Ạ').\
		replace(u'Ằ',u'Ằ').replace(u'Ẵ',u'Ẵ').replace(u'Ẳ',u'Ẳ').replace(u'Ắ',u'Ắ').replace(u'Ặ',u'Ặ').\
		replace(u'Ầ',u'Ầ').replace(u'Ẫ',u'Ẫ').replace(u'Ẩ',u'Ẩ').replace(u'Ấ',u'Ấ').replace(u'Ậ',u'Ậ').\
		replace(u'Ỳ',u'Ỳ').replace(u'Ỹ',u'Ỹ').replace(u'Ỷ',u'Ỷ').replace(u'Ý',u'Ý').replace(u'Ỵ',u'Ỵ').\
		replace(u'Ì',u'Ì').replace(u'Ĩ',u'Ĩ').replace(u'Ỉ',u'Ỉ').replace(u'Í',u'Í').replace(u'Ị',u'Ị').\
		replace(u'Ù',u'Ù').replace(u'Ũ',u'Ũ').replace(u'Ủ',u'Ủ').replace(u'Ú',u'Ú').replace(u'Ụ',u'Ụ').\
		replace(u'Ừ',u'Ừ').replace(u'Ữ',u'Ữ').replace(u'Ử',u'Ử').replace(u'Ứ',u'Ứ').replace(u'Ự',u'Ự').\
		replace(u'È',u'È').replace(u'Ẽ',u'Ẽ').replace(u'Ẻ',u'Ẻ').replace(u'É',u'É').replace(u'Ẹ',u'Ẹ').\
		replace(u'Ề',u'Ề').replace(u'Ễ',u'Ễ').replace(u'Ể',u'Ể').replace(u'Ế',u'Ế').replace(u'Ệ',u'Ệ').\
		replace(u'Ò',u'Ò').replace(u'Õ',u'Õ').replace(u'Ỏ',u'Ỏ').replace(u'Ó',u'Ó').replace(u'Ọ',u'Ọ').\
		replace(u'Ờ',u'Ờ').replace(u'Ỡ',u'Ỡ').replace(u'Ở',u'Ở').replace(u'Ớ',u'Ớ').replace(u'Ợ',u'Ợ').\
		replace(u'Ồ',u'Ồ').replace(u'Ỗ',u'Ỗ').replace(u'Ổ',u'Ổ').replace(u'Ố',u'Ố').replace(u'Ộ',u'Ộ').\
		replace(u'ÒA',u'OÀ').replace(u'ÕA',u'OÃ').replace(u'ỎA',u'OẢ').replace(u'ÓA',u'OÁ').replace(u'ỌA',u'OẠ').\
		replace(u'ÒE',u'OÈ').replace(u'ÕE',u'OẼ').replace(u'ỎE',u'OẺ').replace(u'ÓE',u'OÉ').replace(u'ỌE',u'OẸ').\
		replace(u'ÙY',u'UỲ').replace(u'ŨY',u'UỸ').replace(u'ỦY',u'UỶ').replace(u'ÚY',u'UÝ').replace(u'ỤY',u'UỴ')

def UniStd(str):
	return UniStd_L(UniStd_H(str)).\
		replace(u'òA',u'oà').replace(u'õA',u'oã').replace(u'ỏA',u'oả').replace(u'óA',u'oá').replace(u'ọA',u'oạ').\
		replace(u'òE',u'oè').replace(u'õE',u'oẽ').replace(u'ỏE',u'oẻ').replace(u'óE',u'oé').replace(u'ọE',u'oẹ').\
		replace(u'ùY',u'uỳ').replace(u'ũY',u'uỹ').replace(u'ủY',u'uỷ').replace(u'úY',u'uý').replace(u'ụY',u'uỵ').\
		replace(u'Òa',u'Oà').replace(u'Õa',u'Oã').replace(u'Ỏa',u'Oả').replace(u'Óa',u'Oá').replace(u'Ọa',u'Oạ').\
		replace(u'Òe',u'Oè').replace(u'Õe',u'Oẽ').replace(u'Ỏe',u'Oẻ').replace(u'Óe',u'Oé').replace(u'Ọe',u'Oẹ').\
		replace(u'Ùy',u'Uỳ').replace(u'Ũy',u'Uỹ').replace(u'Ủy',u'Uỷ').replace(u'Úy',u'Uý').replace(u'Ụy',u'Uỵ')

# print(load_file())
dict_abbs = abb_file()
# load_file(dict_abbs, TRAIN_FILE, process_train)
files = {"test": TEST_FILE}
save_path = {"test": process_test}
test_file(dict_abbs, files, save_path)
# print(abb_words)