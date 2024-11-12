library(caret)
library(data.table)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
#library(rminer)
library(nnet)
#install.packages("Amelia")
library(Amelia)
library(caTools)
#install.packages("dummy")
library(dummy)

install.packages("devtools")
library(devtools)
library(dplyr)
library(magrittr)
library(ggplot2)


Credit_Card <- read.csv('/Users/deepanalla/Downloads/Application_Data.csv', header = TRUE)


#PERFORM ONE HOT ENCODING ON GENDER - To differentiate columns for M and F
# Convert Applicant_Gender to factor
Credit_Card$Applicant_Gender <- factor(Credit_Card$Applicant_Gender)
# Perform one-hot encoding for the Applicant_Gender column
encoded_gender <- model.matrix(~ Applicant_Gender - 1, data = Credit_Card)
# Convert the encoded data to a dataframe
encoded_gender <- as.data.frame(encoded_gender)
# Get the names of the levels in the Applicant_Gender column
gender_level_names <- levels(Credit_Card$Applicant_Gender)
gender_level_names
# Modify the column names to include both the original column name and the level name
encoded_gender_colnames <- paste("Applicant_Gender", gender_level_names, sep = "_")
# Set the modified column names for the encoded data
colnames(encoded_gender) <- encoded_gender_colnames
Credit_Card <- cbind(Credit_Card, encoded_gender)

#Min Max Scaling - To scale Income, Working years, Applicant Age
# Define the features to scale
features_to_scale <- c("Total_Income", "Years_of_Working", "Applicant_Age")
# Perform Min-Max scaling
Credit_Card[features_to_scale] <- apply(Credit_Card[features_to_scale], 2, function(x) (x - min(x)) / (max(x) - min(x)))


#To set Levels for Education
# Check unique levels in Education_Type column
unique_levels <- unique(Credit_Card$Education_Type)
# Print unique levels to identify any discrepancies
print(unique_levels)
# Update level_mapping to include all unique levels
level_mapping <- setNames(1:length(unique_levels), unique_levels)
# Apply ordinal encoding to Education_Type column
Credit_Card$Education_Type <- as.integer(factor(Credit_Card$Education_Type, levels = names(level_mapping)))
# Optionally, you can rename the column to indicate that it's ordinal encoded
names(Credit_Card)[names(Credit_Card) == "Education_Type"] <- "Education_Type_Encoded"

#ONE HOT ENCODING - FOR FAMILY STATUS, INCOME TYPE AND HOUSING TYPE
# Columns to encode
columns_to_encode <- c("Family_Status", "Income_Type", "Housing_Type")

# Convert columns to factor with consistent levels
Credit_Card[columns_to_encode] <- lapply(Credit_Card[columns_to_encode], function(x) {
  factor(x, levels = unique(x))
})

# Perform one-hot encoding for the specified columns
encoded_data <- model.matrix(~ . - 1, data = Credit_Card[, columns_to_encode])

# Convert the encoded data to a dataframe
encoded_data <- as.data.frame(encoded_data)

# Set the modified column names for the encoded data
colnames(encoded_data) <- gsub("[.]", "_", colnames(encoded_data))  # Replace '.' in column names with '_'

# Optionally, you can add the encoded columns back to the original dataframe
Credit_Card <- cbind(Credit_Card, encoded_data)

#Now dropping columns. Only keeping columns which are required for Model Training 

# Drop columns by names
# Drop columns by names
columns_to_drop <- c("Applicant_ID", "Applicant_Gender", "Income_Type", "Family_Status", "Housing_Type", "Job_Title", "Total_Bad_Debt", "Total_Good_Debt", "Owned_Work_Phone", "Owned_Mobile_Phone", "Owned_Email")
Credit_Card <- Credit_Card[, !names(Credit_Card) %in% columns_to_drop]


library(caret)
library(ggplot2)
library(pROC)

install.packages("corrplot")
library(corrplot)
install.packages("ggplot2")
install.packages("reshape2")
library(ggplot2)
library(reshape2)

ggplot(data = Credit_Card, aes(x = Years_of_Working, y = Applicant_Age)) +
  geom_point(alpha = 0.6) +  # Adjust point transparency with alpha
  labs(x = "Years of Working", y = "Applicant Age", 
       title = "Years of Working vs Applicant Age") +
  theme_minimal()

ggplot(data = Credit_Card, aes(x = Total_Family_Members, y = Total_Children)) +
  geom_point(alpha = 0.6) +  # Adjust point transparency with alpha
  labs(x = "Total Family Members", y = "Total Children", 
       title = "Total Family Members and Children") +
  theme_minimal()


# Load necessary libraries
library(caret)  # for data splitting and model training
library(Metrics)  # for calculating accuracy metrics
install.packages("Metrics")


# Splitting data into training and testing sets
set.seed(123)  # for reproducibility
index <- createDataPartition(Credit_Card$Status, p=0.8, list=FALSE)
train_data <- Credit_Card[index, ]
test_data <- Credit_Card[-index, ]

# Fitting a Linear Regression Model
model <- lm(Status ~ ., data = train_data)

# Summarize the model
summary(model)

# Making predictions on the test dataset
predictions <- predict(model, test_data)
predictions

# Calculating the accuracy of the model
# Here we use RMSE (Root Mean Squared Error) as a measure of accuracy
rmse_value <- rmse(test_data$Status, predictions)
print(paste("RMSE: ", rmse_value))

# You can also calculate R-squared for the test set
test_data$residuals <- test_data$Status - predictions
rss <- sum(test_data$residuals^2)
tss <- sum((test_data$balance_due - mean(train_data$balance_due))^2)
rsq <- 1 - rss/tss
print(paste("R-squared: ", rsq))


# Load necessary libraries
library(caret)  # for data splitting
library(pROC)   # for AUC calculation


# Fitting a Logistic Regression Model
model1 <- glm(Status ~ ., data = test_data, family = binomial)

# Summary of the model
summary(model1)

model1
# Making predictions on the test dataset
# Predicting probabilities
probabilities <- predict(model, train_data, type = "response")

# Converting probabilities to binary predictions based on a 0.5 threshold
predictions <- ifelse(probabilities > 0.5, 1, 0)
# Examine the range and variety of predicted probabilities
summary(probabilities)

# Check the first few predicted classifications
head(predictions)

# Check the distribution of actual outcomes
table(train_data$Status) 
# Calculating the accuracy of the model
accuracy <- mean(predictions == train_data$Status)
print(paste("Accuracy: ", accuracy))


# Print the column names to check for unexpected spaces
print(names(train_data))

# Remove leading and trailing spaces from column names
names(train_data) <- trimws(names(train_data))
names(test_data) <- trimws(names(test_data))


# Assuming 'df' is your dataframe
# Rename columns in the Credit_Card dataframe
colnames(Credit_Card)[11] <- "Applicant_Gender_F"
colnames(Credit_Card)[12] <- "Applicant_Gender_M"
colnames(Credit_Card)[13] <- "Family_Status_Married"
colnames(Credit_Card)[14] <- "Family_Status_Single_Not_Married"
colnames(Credit_Card)[15] <- "Family_Status_Civil_Marriage"
colnames(Credit_Card)[16] <- "Family_Status_Separated"
colnames(Credit_Card)[17] <- "Family_Status_Widow"
colnames(Credit_Card)[18] <- "Income_Type_Commercial_Associate"
colnames(Credit_Card)[19] <- "Income_Type_State_Servant"
colnames(Credit_Card)[20] <- "Income_Type_Student"
colnames(Credit_Card)[21] <- "Income_Type_Pensioner"
colnames(Credit_Card)[22] <- "Housing_Type_Rented_Apartment"
colnames(Credit_Card)[23] <- "Housing_Type_Municipal_Apartment"
colnames(Credit_Card)[24] <- "Housing_Type_With_Parents"
colnames(Credit_Card)[25] <- "Housing_Type_Co_op_Apartment"
colnames(Credit_Card)[26] <- "Housing_Type_Office_Apartment"


set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(Credit_Card), 0.8 * nrow(Credit_Card))
train <- Credit_Card[train_indices, ]
test <- Credit_Card[-train_indices, ]
# Fit the model
# Assume 'default' is the response variable and it's a binary classification problem
rf_model <- randomForest(Status ~ ., data = train, ntree = 100)
rf_model
summary(rf_model)

# Make predictions
predictions <- predict(rf_model, test)

# Evaluate the model
confusionMatrix <- table(test$Status, predictions)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
print(paste("Accuracy:", accuracy))



