import os, sys
import datetime
import functools


def get_date_str():
	datetim = str(datetime.datetime.now()).replace('.', '').replace('-', '').replace(':', '').replace(' ', '')[2:14]
	return datetim


class WriteLog(object):
	def __init__(self, dest):
		if os.path.isfile(dest):
			name = dest.split('.')[0:-1]
			ext = dest.split('.')[-1]
			new_name = name+get_date_str()+'.'+ext
			with open(new_name, 'w') as f:
				f.write(open(dest, 'r').read())
		self.logfile = dest
		# print 'logfile:', logfile
		self.f = open(dest,'w')
		self.f.close()

	def fprint(self, *stt):
		sto = functools.reduce(lambda x, y: str(x)+' '+str(y), list(stt))
		print(sto)
		try:
			sto = str(datetime.datetime.now())+':'+ sto
		except:
			pass
		assert os.path.exists(self.logfile)
		self.f = open(self.logfile, 'a')
		try:
			self.f.write('\n'+sto)
		except:
			pass
		self.f.close()

	def clear(self):
		self.f = open(self.logfile, 'w')
		self.f.close()

	def close(self):
		# print ('no need to close')
		self.f.close()
		return