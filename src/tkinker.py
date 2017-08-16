import matplotlib
matplotlib.use('TkAgg')
from numpy import *
from tkinter.tix import *
from src.reg_trees.book.tree_reg_classifier import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class Tkinker:

    def __init__(self, dataset: matrix):
        root = Tk()

        self.reDraw.f = Figure(figsize=(5, 4), dpi=100)
        self.reDraw.canvas = FigureCanvasTkAgg(self.reDraw.f, master=root)
        self.reDraw.canvas.show()
        self.reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

        Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
        Label(root, text="tolN").grid(row=1, column=0)
        tolNentry = Entry(root)
        tolNentry.grid(row=1, column=1)
        tolNentry.insert(0, '10')
        Label(root, text="tolS").grid(row=2, column=0)
        tolSentry = Entry(root)
        tolSentry.grid(row=2, column=1)
        tolSentry.insert(0, '1.0')
        Button(root, text="ReDraw", command=self.drawNewTree).grid(row=1, column=2, rowspan=3)
        chkBtnVar = IntVar()
        chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
        chkBtn.grid(row=3, column=0, columnspan=2)
        self.reDraw.rawDat = mat(loadDataSet('../resources/trees/bikeSpeedVsIq_train.txt'))
        self.reDraw.testDat = arange(min(self.reDraw.rawDat[:, 0]), max(self.reDraw.rawDat[:, 0]), 0.01)
        self.reDraw(1.0, 10)
        root.mainloop()

        Button(root, text='Quit', fg="black", command=root.quit).grid(row=1, column=2)

    def reDraw(self, tolS, tolN):
        self.reDraw.f.clf()
        self.reDraw.a = self.reDraw.f.add_subplot(111)
        if self.chkBtnVar.get():
            if tolN < 2: tolN = 2
            myTree = createTree(self.reDraw.rawDat, modelLeaf, modelErr, (tolS, tolN))
            yHat = createForeCast(myTree, self.reDraw.testDat, modelTreeEval)
        else:
            myTree = createTree(self.reDraw.rawDat, ops=(tolS, tolN))
            yHat = createForeCast(myTree, self.reDraw.testDat)
        rawData = self.reDraw.rawDat
        X = [x[0] for x in rawData[:, 0].tolist()]
        Y = [y[0] for y in rawData[:, 1].tolist()]
        self.reDraw.a.scatter(X, Y, s=5)
        self.reDraw.a.plot(self.reDraw.testDat, yHat, linewidth=2.0)
        self.reDraw.canvas.show()

    def getInputs(self):
        try:
            tolN = int(self.tolNentry.get())
        except:
            tolN = 10
        print("enter Integer for tolN")
        self.tolNentry.delete(0, END)
        self.tolNentry.insert(0, '10')
        try:
            tolS = float(self.tolSentry.get())
        except:
            tolS = 1.0
        print("enter Float for tolS")
        self.tolSentry.delete(0, END)
        self.tolSentry.insert(0, '1.0')
        return tolN, tolS

    def drawNewTree(self):
        tolN, tolS = self.getInputs()
        self.reDraw(tolS, tolN)