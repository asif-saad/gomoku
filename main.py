
#----------------------------------------------------------------------------
# Created By  : Asif Saad, Shihab Newaz
# Created Date: 26.05.2023
# ---------------------------------------------------------------------------
# This is a simple GUI in PyQT for MakeC engine from MakeC.py

from PyQt5 import QtCore, QtGui, QtWidgets
from MakeC import MakeCEngine


class QLabel_alterada(QtWidgets.QLabel):
    clicked=QtCore.pyqtSignal()

    def mousePressEvent(self, ev):
        self.clicked.emit()

class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        QtWidgets.QWidget.__init__(self)
        self.pixmap_black = QtGui.QPixmap('black.jpg')
        self.pixmap_white = QtGui.QPixmap('white.jpg')
        self.user_turn = 0
        self.ai_turn = 1
        self.ai_depth = 2
        self.move_hint = (7, 7) #initial move hint
        self.setWindowTitle('MakeC')
        self.setWindowIcon(QtGui.QIcon('MakeC.ico'))
        self.game_over = False
        self.empty_pixmap = QtGui.QPixmap('empty.jpg')
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        self.labels = {}
        self.game_engine = MakeCEngine()
        self.move_count = 0
        for row in range(15):
            for column in range(15):
                label = QLabel_alterada(self)
                label.coord_ = (row,column)
                label.setObjectName(str(row)+'-'+str(column))
                #label.setPixmap(self.empty_pixmap)
                self.layout.addWidget(label, row, column)
                label.clicked.connect(self.processUserMove) 
                self.labels[label.coord_] = label
        self.drawEmptyBoard()
    
    
    def drawEmptyBoard(self):
        for key in self.labels:
            self.labels[key].setPixmap(self.empty_pixmap)


    def processUserMove(self):
        label_sender = self.sender()
        user_move = label_sender.coord_
        winner_message = ''
        if (not self.game_over) and self.game_engine.can_move(user_move):
            label_sender.setPixmap(self.pixmap_black)
            last_eval = self.game_engine.user_move(self.user_turn, user_move)
            self.move_count += 1
            self.move_hint = user_move
            if last_eval == float('inf'):
                winner_message = 'User wins'
                self.game_over = True
                self.restart_dialogue(winner_message)
                return
            if self.move_count == 225:
                winner_message = 'Tie'
                self.game_over = True
                self.restart_dialogue(winner_message)
                return
            if not self.game_over:
                last_eval, last_ai_move = self.game_engine.ai_move(self.ai_depth, self.ai_turn, self.move_hint)
                self.move_count += 1
                self.labels[last_ai_move].setPixmap(self.pixmap_white)
                if last_eval == -1 * float('inf'):
                    winner_message = 'AI wins'
                    self.game_over = True
                    self.restart_dialogue(winner_message)
                    return
    

    def restart_dialogue(self, winner_message):
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(winner_message)
        dlg.setText("Play again?")
        dlg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        dlg.setIcon(QtWidgets.QMessageBox.Question)
        button = dlg.exec()

        if button == QtWidgets.QMessageBox.Yes:
            self.game_engine.reset_board()
            self.drawEmptyBoard()
            self.game_over = False
            self.move_count = 0
            dlg.close()
        else:
            dlg.close()
            self.close()
        

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(100, 100, 700, 700)
    window.setFixedSize(700, 700)
    window.setStyleSheet('background-color: white; border: 0.5px solid  black')
    window.show()
    sys.exit(app.exec_())