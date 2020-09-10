"""
Task Environment class to manage tasks in the scene.
"""

import math


class TaskEnvironment:
    def __init__(self, sim_client, objectsID, traysID):
        self.objectsID = objectsID
        self.traysID = traysID
        self.sim_client = sim_client
        self.finishedObjects = []

    def update(self):
        """
        Update if a new task is finished.
        """
        for trayID in self.traysID:
            for elem in self.objectsID:
                if not (elem in set(self.finishedObjects)):
                    # If any object is put in tray
                    touch = self.sim_client.getContactPoints(
                        self.objectsID[elem], trayID)
                    if touch:
                        self.finishedObjects.append(elem)
                    # If the object is placed above other objects in tray
                    else:
                        for finished in self.finishedObjects:
                            touch = self.sim_client.getContactPoints(
                                self.objectsID[elem], self.objectsID[finished])
                            if touch:
                                self.finishedObjects.append(elem)

    def isDone(self):
        """
        Check if all tasks are finished.
        """
        if len(self.objectsID) == len(self.finishedObjects):
            return True
        else:
            return False

    def checkFinishedObject(self, objectName):
        for elem in self.finishedObjects:
            if elem == objectName:
                return True
        return False

    def totalFinishedObjects(self):
        return len(self.finishedObjects)
