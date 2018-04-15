import utils


class Essay():
	def __init__(self, filepath, prompt):
		self.filepath = filepath
		self.prompt = prompt
		self.data = utils.open_file_read(self.filepath)       