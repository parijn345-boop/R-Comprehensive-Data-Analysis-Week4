# Week 4: Integrated Data Cleaning, Visualization, Statistical Analysis and Modeling
# Base R only; uses the built-in iris dataset.
data("iris")
df <- iris

# 1. Data overview and quality checks
cat("Dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
str(df); summary(df)
cat("Missing values by column:\n"); print(colSums(is.na(df)))
cat("Duplicate rows:", sum(duplicated(df)), "\n")
stopifnot(nrow(df) == 150, ncol(df) == 5)

# 2. Descriptive statistics and charts
num <- df[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")]
cat("Correlation matrix:\n"); print(round(cor(num), 3))
png("week4_hist_petallength.png", width=1000, height=650, res=130)
hist(df$Petal.Length, main="Petal Length Distribution", xlab="Petal length (cm)", col="lightblue"); dev.off()
png("week4_boxplot_species.png", width=1000, height=650, res=130)
boxplot(Petal.Length ~ Species, data=df, main="Petal Length by Species", xlab="Species", ylab="Petal length (cm)", col="lightblue"); dev.off()
png("week4_scatter_sepal_petal.png", width=1000, height=650, res=130)
plot(df$Sepal.Length, df$Petal.Length, pch=19, xlab="Sepal length (cm)", ylab="Petal length (cm)", main="Sepal Length vs Petal Length", col=as.integer(df$Species))
legend("topleft", legend=levels(df$Species), col=1:3, pch=19); dev.off()
png("week4_pairs.png", width=1000, height=1000, res=130)
pairs(num, main="Pairwise Relationships Among Measurements"); dev.off()

# 3. Correlation and hypothesis testing
cat("Pearson correlation test:\n"); print(cor.test(df$Sepal.Length, df$Petal.Length))
cat("ANOVA for petal length by species:\n")
anova_fit <- aov(Petal.Length ~ Species, data=df); print(summary(anova_fit))
cat("Tukey post-hoc comparisons:\n"); print(TukeyHSD(anova_fit))

# 4. Multiple linear regression
model <- lm(Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Species, data=df)
cat("Regression summary:\n"); print(summary(model))

# 5. Holdout validation
set.seed(42)
train_idx <- sample(seq_len(nrow(df)), size=floor(0.8*nrow(df)))
train <- df[train_idx, ]; test <- df[-train_idx, ]
hold_model <- lm(Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Species, data=train)
hold_pred <- predict(hold_model, newdata=test)
hold_rmse <- sqrt(mean((test$Petal.Length-hold_pred)^2))
hold_mae <- mean(abs(test$Petal.Length-hold_pred))
hold_r2 <- 1-sum((test$Petal.Length-hold_pred)^2)/sum((test$Petal.Length-mean(test$Petal.Length))^2)
cat("Holdout RMSE:", hold_rmse, " MAE:", hold_mae, " R-squared:", hold_r2, "\n")

# 6. Five-fold cross-validation
set.seed(42)
fold_id <- sample(rep(1:5, length.out=nrow(df)))
cv <- data.frame(Fold=1:5, RMSE=NA_real_, MAE=NA_real_, R2=NA_real_)
for (k in 1:5) {
  tr <- df[fold_id != k, ]; te <- df[fold_id == k, ]
  fit <- lm(Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Species, data=tr)
  pr <- predict(fit, newdata=te)
  cv$RMSE[k] <- sqrt(mean((te$Petal.Length-pr)^2))
  cv$MAE[k] <- mean(abs(te$Petal.Length-pr))
  cv$R2[k] <- 1-sum((te$Petal.Length-pr)^2)/sum((te$Petal.Length-mean(te$Petal.Length))^2)
}
cat("Five-fold CV metrics:\n"); print(cv)
cat("Mean CV metrics:\n"); print(colMeans(cv[, c("RMSE", "MAE", "R2")]))
write.csv(cv, "week4_cv_metrics.csv", row.names=FALSE)

# 7. Regression diagnostics
png("week4_regression_diagnostics.png", width=1000, height=1000, res=130)
par(mfrow=c(2,2)); plot(model); par(mfrow=c(1,1)); dev.off()
cat("Shapiro-Wilk test on model residuals:\n"); print(shapiro.test(residuals(model)))
cat("Saved week4 chart PNGs and week4_cv_metrics.csv in the working directory.\n")
