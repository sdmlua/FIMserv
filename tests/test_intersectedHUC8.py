import fimserve as fm

user_boundary = "/Users/supath/Downloads/MSResearch/FIMpef/CodeUsage/SampleData/Data/Neuse/FIMEvaluatedExtent.shp"

summary = fm.getIntersectedHUC8ID(user_boundary)
print(summary)
