import utils


class Essay():
	def __init__(self, filepath, prompt):
		self.filepath = filepath
		self.prompt = prompt                                 #prompt of the essay
		self.data = utils.open_file_read(self.filepath)      #content of the essay 