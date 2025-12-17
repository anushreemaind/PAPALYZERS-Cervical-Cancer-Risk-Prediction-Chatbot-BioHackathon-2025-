rm(list=ls())
if (!require("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!require("factoextra", quietly = TRUE)) install.packages("factoextra")
if (!require("dendextend", quietly = TRUE)) install.packages("dendextend")
if (!require("flexclust", quietly = TRUE)) install.packages("flexclust")
if (!require("caret", quietly = TRUE)) install.packages("caret")
if (!require("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!require("glmnet", quietly = TRUE)) install.packages("glmnet")
if (!require("smotefamily", quietly = TRUE)) install.packages("smotefamily")

# Load necessary libraries
library(tidyverse)
library(caret)
library(smotefamily)
library(cluster)
library(randomForest)
library(glmnet)
library(rpart)
library(rpart.plot)

# 1. Data loading and preprocessing
cervical_data <- read.csv("~/Desktop/cervical-cancer_csv.csv", na.strings = "?")
cervical_data <- cervical_data %>%
  mutate(across(everything(), as.numeric)) %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 2. Address class imbalance using smotefamily
set.seed(123)

# Prepare data for SMOTE - excluding other target variables and specified columns
columns_to_exclude <- c("Dx", "Dx.Cancer", "Dx.CIN", "Dx.HPV", "Hinselmann", "Schiller", "Citology")
features <- cervical_data %>% select(-Biopsy, -all_of(columns_to_exclude))
target <- cervical_data$Biopsy

# Apply SMOTE using smotefamily
smote_result <- SMOTE(X = features, target = target, K = 5, dup_size = 2)

# Extract balanced dataset
cervical_balanced <- as.data.frame(smote_result$data)
colnames(cervical_balanced)[ncol(cervical_balanced)] <- "Biopsy"

# Check class distribution
print("Class distribution after SMOTE:")
print(table(cervical_balanced$Biopsy))

# 3. Feature selection using LASSO - Graphical output
x <- as.matrix(cervical_balanced %>% select(-Biopsy))
y <- cervical_balanced$Biopsy

# Get the full LASSO path
lasso_model <- glmnet(x, y, alpha = 1, family = "binomial")

# Get coefficients at optimal lambda
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
optimal_lambda <- cv.lasso$lambda.min

# Extract coefficients and sort by absolute value
lasso_coef <- coef(lasso_model, s = optimal_lambda)[,1]
lasso_coef <- lasso_coef[-1]  # Remove intercept

# Create data frame for visualization
importance_df <- data.frame(
  Feature = names(lasso_coef),
  Coefficient = lasso_coef,
  Absolute_Importance = abs(lasso_coef)
) %>%
  arrange(desc(Absolute_Importance)) %>%
  head(10)  # Top 10 features

# Create the plot
ggplot(importance_df, aes(x = reorder(Feature, Absolute_Importance), y = Absolute_Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  ggtitle("Top 10 Feature Importance by LASSO Coefficient Magnitude") +
  xlab("Features") +
  ylab("Importance (Absolute Coefficient Value)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 10)
  )

# Print the optimal lambda value
cat("Optimal lambda value:", round(optimal_lambda, 6), "\n")
cat("Number of features with non-zero coefficients:", sum(lasso_coef != 0), "\n")
cat("Top 10 features selected:\n")
print(importance_df[, c("Feature", "Coefficient")])

# Store the selected feature names for later use
feature_names <- as.character(importance_df$Feature)

# 4. Data splitting
set.seed(123)
trainIndex <- createDataPartition(cervical_balanced$Biopsy, p = 0.8, list = FALSE)
train_data <- cervical_balanced[trainIndex, ]
test_data <- cervical_balanced[-trainIndex, ]

# 5. Model training
rf_model <- randomForest(as.factor(Biopsy) ~ ., data = train_data, importance = TRUE)

# 6. Prediction
rf_pred <- predict(rf_model, test_data)

# 7. Evaluation with better labels
rf_pred_labels <- factor(ifelse(rf_pred == 1, "Positive", "Negative"), 
                         levels = c("Positive", "Negative"))
test_labels <- factor(ifelse(test_data$Biopsy == 1, "Positive", "Negative"), 
                      levels = c("Positive", "Negative"))

# 8. Feature importance plot (Mean Decrease Accuracy)
importance_df <- data.frame(
  Feature = rownames(rf_model$importance),
  Importance = rf_model$importance[, "MeanDecreaseAccuracy"]
) %>% 
  arrange(desc(Importance)) %>% 
  head(10)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  ggtitle("Top 10 Feature Importance for Cervical Cancer Prediction") +
  xlab("Features") +
  ylab("Importance (Mean Decrease Accuracy)") +
  theme_minimal()

