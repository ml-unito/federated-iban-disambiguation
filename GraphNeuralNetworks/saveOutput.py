from datetime import datetime

class SaveOutput():
  """ Simple class to save data on file """

  def __init__(self, filePath, fileName=None, debug=False):
    self.debug = debug
    if not self.debug:
      if fileName: self.fileName = fileName
      else: self.fileName = "log_" + str(datetime.now()).split(".")[0].replace(" ", "_") + ".txt"
      self.filePath = filePath + self.fileName

  def __call__(self, text, printAll=True):
    if not self.debug:
      with open(self.filePath, "a") as file:
        file.write(str(text))
        file.write("\n")
    if (printAll):
      print(str(text))