import pandas as pd

df = pd.read_excel("/Users/mohamedibrahim/Documents/study/110A/TableauSalesData.xlsx")

df.to_csv("/Users/mohamedibrahim/Documents/study/110A/TableauSalesData.csv", index=False)

print("Done! CSV created.")