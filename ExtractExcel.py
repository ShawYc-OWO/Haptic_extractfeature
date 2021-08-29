import xlrd


class ExelUtil(object):

    def __init__(self, excel_path=None, index=None):
        if excel_path == None:
            excel_path =r"D:\机器人与计算机类的渐进学习\2020-2021 IC专业机器人学习\Project_DataDrivenHaptic\Third Term\Test data\test_in_1cm_circle\Prepocessing\Pushing"
            r"\Train"
        if index == None:
            index = 0
        self.data = xlrd.open_workbook(excel_path)
        self.table = self.data.sheets()[index]
        # 行数
        self.rows = self.table.nrows
        #[[],[],[]]

    def get_data(self):
        result = []
        for i in range(self.rows):
            col = self.table.row_values(i)

            print(col)
            result.append(col)


if __name__ == '__main__':
    ec = ExelUtil()
    ec.get_data()
