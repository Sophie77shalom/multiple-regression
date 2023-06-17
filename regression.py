import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import t

 
df = pd.read_csv('nbaallelo.csv')

 
df = df[['pts', 'elo_i', 'elo_n', 'win_equiv', 'game_location', 'game_result']]

predictor = 'pts'
target = 'win_equiv'

 
plt.scatter(df[predictor], df[target])
plt.xlabel(predictor)
plt.ylabel(target)
plt.title('Simple Linear Regression - Scatterplot')
plt.show()

 
correlation = df[predictor].corr(df[target])
print("Correlation coefficient:", correlation)

 
X = df[[predictor]]
y = df[target]
regression_model = LinearRegression()
regression_model.fit(X, y)
y_predicted = regression_model.predict(X)
r2 = r2_score(y, y_predicted)
print("R-squared value:", r2)

 
n = len(df)
dof = n - 2   
t_critical = t.ppf(0.975, dof)   
se = (1 - correlation ** 2) ** 0.5 / (n - 2) ** 0.5   
test_statistic = correlation / se
p_value = 2 * (1 - t.cdf(abs(test_statistic), dof))

 
print("Test Statistic:", test_statistic)
print("P-value:", p_value)

 
 
predictors = ['pts', 'elo_i', 'elo_n']
target = 'win_equiv'

 
sns.pairplot(df[predictors + [target]])
plt.show()

 
correlation_matrix = df[predictors + [target]].corr()
print("Correlation matrix:")
print(correlation_matrix)

 
X = df[predictors]
y = df[target]
regression_model = LinearRegression()
regression_model.fit(X, y)
y_predicted = regression_model.predict(X)
r2 = r2_score(y, y_predicted)
print("R-squared value:", r2)