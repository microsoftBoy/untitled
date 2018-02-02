import xlwt


def write():
    work_book = xlwt.Workbook()
    work_sheet = work_book.add_sheet('啊哈哈')
    work_sheet.write(0, 0, '帅帅帅')
    work_book.save('pythonForWrite.xls')


if __name__ == '__main__':
    write()
