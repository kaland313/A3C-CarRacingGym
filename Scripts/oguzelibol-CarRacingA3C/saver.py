#!/usr/bin/python
import tensorflow as tf
import os, time, json
from savedata import SaveData
import constants as Constants

class Saver:
    def __init__(self, checkpointDir=Constants.CHECKPOINT_DIR, saveFile=Constants.SAVE_FILENAME, lastSaveTime=time.time()):
        self.saver          =  tf.train.Saver()
        self.data           =  SaveData()
        self.checkpoint     =  tf.train.get_checkpoint_state(checkpointDir)
        self.saveFilename   =  saveFile
        self.lastSaveTime   =  lastSaveTime
        self.lastSaveScore  =  0
        self.checkpointDir  =  checkpointDir


    def saveScores(self, scores):
        with open(self.saveFilename, 'w') as f:
            f.write(json.dumps(scores))

    def loadScores(self):
        try:
            with open(self.saveFilename) as f:
                self.data.setScores(json.loads(f.read().strip()))
                print('Loaded scores, last 5 shown: ', [score['value'] for score in self.data.scores[-5:]])
        except Exception:
            print('Unable to load scores')

    def load(self, session):
        self.loadScores()

        if self.checkpoint is not None and self.checkpoint.model_checkpoint_path is not None:

            # Load checkpoint
            self.saver.restore(session, self.checkpoint.model_checkpoint_path)
            self.data.global_t = int(self.checkpoint.model_checkpoint_path.split("-")[1])

            # Set Wall time
            wall_t_fname = self.checkpointDir + '/' + 'wall_t.' + str(self.data.global_t)
            with open(wall_t_fname, 'r') as f:
                self.data.wall_t = float(f.read())

            print("Checkpoint:", self.checkpoint.model_checkpoint_path, "\nGlobal Step:", self.data.global_t)



    def save(self, session):
        if not os.path.exists(self.checkpointDir):
            os.mkdir(self.checkpointDir)

        # write wall time
        wall_t = time.time() - self.data.start_time
        wall_t_filename = self.checkpointDir + '/' + 'wall_t.' + str(self.data.global_t)
        with open(wall_t_filename, 'w') as f:
            f.write(str(wall_t))

        self.saveScores(self.data.scores)

        # Save TF checkpoint
        self.saver.save(session, self.checkpointDir + '/' + 'checkpoint', global_step = self.data.global_t)

    def canSave(self):
        return time.time() - self.lastSaveTime >= Constants.SAVE_INTERVAL or \
                abs(self.data.scores[-1]['value'] - self.lastSaveScore) >= 40

    def saveIfRequested(self, session):
        if self.data.saveRequested:
            self.data.saveRequested = False
            if self.canSave():
                print("Saving checkpoint as score crossed threshold of:", Constants.SAVE_SCORE_THRESHOLD)
                self.save(session)
                self.lastSaveTime = time.time()
