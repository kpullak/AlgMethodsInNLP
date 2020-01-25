import xlrd

loc = ("/Users/chaitu/Desktop/NCSU_Courses/OR_506/Project/StockPrices.xlsx")

wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
sheet.cell_value(0, 0) 
  
# Extracting number of rows 
#print(sheet.nrows)

arrayofvalues = sheet.col_values(0)
print(len(arrayofvalues))

print(arrayofvalues[:100])