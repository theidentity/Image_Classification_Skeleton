from glob import glob
import os


def getCounts(path):
	for root,dirs,files in os.walk(path):
		print root
		print len(files)

path = '.'
getCounts(path)