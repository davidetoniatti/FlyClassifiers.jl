using MLUtils: MLUtils, flatten, mapobs, splitobs
using MLDatasets, BenchmarkTools
using MultivariateStats, StatsBase
using FlyClassifiers
using Random

# Data Preparation
X, y = MLDatasets.FashionMNIST()[:]
X = MLUtils.flatten(X)

mask = map(y -> y in [2,5], y)
X = X[:, mask]
y = y[mask]

# Standardization
dt = StatsBase.fit(ZScoreTransform, X; dims=2)
X = StatsBase.transform(dt, X)
replace!(X, NaN => 0)

# Split train/test (80% - 20%)
train, test = splitobs((X, y); at = 0.8, shuffle = false)
X_train, y_train = train
X_test, y_test = test;

# PCA
M = MultivariateStats.fit(PCA, X_train; maxoutdim=20)
X_train = MultivariateStats.predict(M, X_train)
X_test = MultivariateStats.predict(M, X_test)

# Training
seed = 42

d = size(X_train)[1]
m = 50_000 # projection dimensionality

s = 5

k = 128
γ = 0.9

accuracy_score(y_true, y_pred) = round(100 * mean(y_pred .== y_true); digits=2)

B = RandomBinaryProjectionMatrix(m, d, s; seed)
U = RandomUniformProjectionMatrix(m, d; seed)

@btime FlyClassifiers.fit(FlyNNM, X_train, y_train, B, k, γ)

# model1 = FlyClassifiers.fit(FlyNNM, X_train, y_train, B, k, γ)
# model2 = FlyClassifiers.fit(FlyNNM, X_train, y_train, U, k, γ)
# model3 = FlyClassifiers.fit(FlyNNA, X_train, y_train, U, k)
# model4 = FlyClassifiers.fit(FlyNNA, X_train, y_train, B, k)
#
# y_pred = FlyClassifiers.predict(model1, X_test)
# println(accuracy_score(y_test, y_pred))
#
# y_pred = FlyClassifiers.predict(model2, X_test)
# println(accuracy_score(y_test, y_pred))
#
# y_pred = FlyClassifiers.predict(model3, X_test)
# println(accuracy_score(y_test, y_pred))
#
# y_pred = FlyClassifiers.predict(model4, X_test)
# println(accuracy_score(y_test, y_pred))
