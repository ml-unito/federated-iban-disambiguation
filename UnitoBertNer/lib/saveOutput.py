from datetime import datetime

class SaveOutput():
  """ Simple class to save data on file """

  def __init__(self, filePath, fileName=None, printAll=False):
    
    if fileName: self.fileName = fileName
    else: 
      self.fileName = "log_" + str(datetime.now()).split(".")[0].replace(" ", "_") + ".txt"
      self.fileName = self.fileName.replace(":", "_")
    self.filePath = filePath + self.fileName
    self.printAll = printAll

  def __call__(self, text):
    with open(self.filePath, "a") as file:
      file.write(text)
      file.write("\n")
    if (self.printAll):
      print(text)