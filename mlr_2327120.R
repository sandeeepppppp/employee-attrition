# Load necessary libraries
library(dplyr)

# Read the dataset
hrm_data <- read.csv("hrm.csv")

# Handle missing values by replacing them with the mean of the respective columns
hrm_data <- hrm_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Build the multiple regression model
model <- lm(Attrition ~ Education + PercentSalaryHike + EmployeeNumber + EnvironmentSatisfaction + HourlyRate + JobInvolvement + JobLevel + MonthlyIncome, data = hrm_data)

# Summary of the model
summary(model)
# Evaluate the model's R-squared
rsquared <- summary(model)$r.squared
cat("R-squared:", rsquared, "\n")

# Predicting on the training data
predictions <- predict(model, newdata = hrm_data)

# Calculating RMSE
rmse <- sqrt(mean((hrm_data$Attrition - predictions)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")


#################################################################################################

# Load necessary library for splitting data
library(caret)

# Set seed for reproducibility
set.seed(123)

# Read the dataset
hrm_data <- read.csv("hrm.csv")

# Handle missing values by replacing them with the mean of the respective columns
hrm_data <- hrm_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Split data into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(hrm_data$Attrition, p = 0.7, list = FALSE)
train_data <- hrm_data[trainIndex, ]
test_data <- hrm_data[-trainIndex, ]

########################################################################
# Load necessary library for Lasso regression

install.packages("glmnet")
library(glmnet)

# Convert data into matrix format required by glmnet
x_train <- as.matrix(train_data[, c("Education", "EmployeeCount", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "MonthlyIncome")])
y_train <- train_data$Attrition

# Fit Lasso regression model
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)  # alpha = 1 for Lasso

# Print optimal lambda value chosen by cross-validation
cat("Optimal lambda for Lasso:", lasso_model$lambda.min, "\n")

# Predict on test data
x_test <- as.matrix(test_data[, c("Education", "EmployeeCount", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "MonthlyIncome")])
lasso_pred <- predict(lasso_model, newx = x_test, s = "lambda.min")

# Calculate RMSE for Lasso
lasso_rmse <- sqrt(mean((test_data$Attrition - lasso_pred)^2))
cat("Lasso Regression RMSE:", lasso_rmse, "\n")



# Optionally, if you want to print R-squared for Lasso
lasso_r_squared <- cor(test_data$Attrition, lasso_pred)^2
cat("Lasso Regression R-squared:", lasso_r_squared, "\n")



#########################################################################################
 #ridge
# Fit Ridge regression model
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)  # alpha = 0 for Ridge


# Print optimal lambda value chosen by cross-validation
cat("Optimal lambda for Ridge:", ridge_model$lambda.min, "\n")

# Predict on test data
ridge_pred <- predict(ridge_model, newx = x_test, s = "lambda.min")

# Calculate RMSE for Ridge
ridge_rmse <- sqrt(mean((test_data$Attrition - ridge_pred)^2))
cat("Ridge Regression RMSE:", ridge_rmse, "\n")
#Optionally, if you want to print R-squared for Ridge
ridge_r_squared <- cor(test_data$Attrition, ridge_pred)^2
cat("Ridge Regression R-squared:", ridge_r_squared, "\n")

