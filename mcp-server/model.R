# Business Opportunity Score Model
# This R script builds a simple ML model using census data to predict
# business opportunity scores based on state, corporation type, and employee size.

library(tidyverse)
library(tidymodels)

# Read the census data
census <- read_csv("census.csv", show_col_types = FALSE)

# Clean column names
colnames(census) <- c(
  "state", "naics_code", "naics_label", "corp_type",
  "emp_size", "year", "establishments", "annual_payroll",
  "q1_payroll", "employees"
)

# Filter to relevant data:
# - Only "Total for all sectors" (NAICS code 00)
# - Exclude "All establishments" aggregates
# - Exclude territories for cleaner demo
census_clean <- census %>%
  filter(
    naics_code == "00",
    corp_type != "All establishments",
    emp_size != "All establishments",
    !state %in% c("American Samoa", "Guam", "Puerto Rico",
                  "United States Virgin Islands",
                  "Commonwealth of the Northern Mariana Islands")
  ) %>%
  mutate(
    # Convert payroll columns (remove commas and convert to numeric)
    annual_payroll = as.numeric(gsub(",", "", annual_payroll)),
    q1_payroll = as.numeric(gsub(",", "", q1_payroll)),
    establishments = as.numeric(gsub(",", "", establishments)),
    employees = as.numeric(gsub(",", "", employees))
  ) %>%
  filter(!is.na(annual_payroll) & !is.na(employees) & employees > 0)

# Create derived features for the model
census_features <- census_clean %>%
  mutate(
    # Average salary per employee (in thousands)
    avg_salary = annual_payroll / employees,
    # Employees per establishment
    emp_per_estab = employees / establishments,
    # Payroll growth indicator (Q1 annualized vs actual)
    payroll_momentum = (q1_payroll * 4) / annual_payroll
  ) %>%
  filter(!is.na(avg_salary) & !is.na(emp_per_estab) & !is.na(payroll_momentum))

# Create a "Business Opportunity Score" (0-100)
# This combines multiple factors into a single score
census_features <- census_features %>%
  group_by(corp_type, emp_size) %>%
  mutate(
    # Normalize salary within group (higher is better)
    salary_score = (avg_salary - min(avg_salary)) / (max(avg_salary) - min(avg_salary) + 0.01),
    # Normalize momentum (closer to 1 is stable, higher suggests growth)
    momentum_score = pmin(payroll_momentum, 1.5) / 1.5,
    # Establishment density score
    density_score = (log(establishments + 1) - min(log(establishments + 1))) /
                    (max(log(establishments + 1)) - min(log(establishments + 1)) + 0.01)
  ) %>%
  ungroup() %>%
  mutate(
    # Combine into final score (weighted average)
    opportunity_score = round(
      (salary_score * 0.4 + momentum_score * 0.35 + density_score * 0.25) * 100,
      1
    ),
    # Ensure score is bounded
    opportunity_score = pmin(pmax(opportunity_score, 0), 100)
  )

# Prepare model data
model_data <- census_features %>%
  select(state, corp_type, emp_size, opportunity_score,
         avg_salary, emp_per_estab, establishments, employees) %>%
  mutate(
    state = as.factor(state),
    corp_type = as.factor(corp_type),
    emp_size = as.factor(emp_size)
  )

# Add log-transformed establishments for the model
model_data <- model_data %>%
  mutate(log_establishments = log(establishments + 1))

# Define and fit Random Forest model using tidymodels
rf_model <- rand_forest(trees = 100) %>%
  set_engine("ranger", importance = "impurity", seed = 42) %>%
  set_mode("regression") %>%
  fit(opportunity_score ~ state + corp_type + emp_size + avg_salary + emp_per_estab + log_establishments,
      data = model_data)

# Print model summary
cat("=== Random Forest Model Summary ===\n")
print(rf_model)

# Generate predictions
predictions <- model_data %>%
  mutate(.pred = predict(rf_model, new_data = model_data)$.pred)

# Create the lookup table
lookup_table <- predictions %>%
  mutate(
    predicted_score = round(.pred, 1),
    # Add confidence based on sample size
    confidence = case_when(
      establishments >= 1000 ~ "high",
      establishments >= 100 ~ "medium",
      TRUE ~ "low"
    )
  ) %>%
  select(state, corp_type, emp_size, predicted_score, confidence,
         establishments, employees, avg_salary) %>%
  arrange(state, corp_type, emp_size)

# Simplify employee size labels for easier API usage
lookup_table <- lookup_table %>%
  mutate(
    emp_size_code = case_when(
      str_detect(emp_size, "less than 5") ~ "1-4",
      str_detect(emp_size, "5 to 9") ~ "5-9",
      str_detect(emp_size, "10 to 19") ~ "10-19",
      str_detect(emp_size, "20 to 49") ~ "20-49",
      str_detect(emp_size, "50 to 99") ~ "50-99",
      str_detect(emp_size, "100 to 249") ~ "100-249",
      str_detect(emp_size, "250 to 499") ~ "250-499",
      str_detect(emp_size, "500 to 999") ~ "500-999",
      str_detect(emp_size, "1,000") ~ "1000+",
      TRUE ~ as.character(emp_size)
    ),
    corp_type_code = case_when(
      str_detect(corp_type, "C-corp") ~ "c-corp",
      str_detect(corp_type, "S-corp") ~ "s-corp",
      str_detect(corp_type, "Individual") ~ "sole-proprietor",
      str_detect(corp_type, "Partner") ~ "partnership",
      str_detect(corp_type, "Non-profit") ~ "nonprofit",
      str_detect(corp_type, "Government") ~ "government",
      str_detect(corp_type, "Other noncorp") ~ "other",
      TRUE ~ tolower(gsub(" ", "-", corp_type))
    )
  )

# Write the lookup table
write_csv(
  lookup_table %>%
    select(
      state,
      corp_type = corp_type_code,
      emp_size = emp_size_code,
      score = predicted_score,
      confidence,
      establishments,
      employees,
      avg_salary_thousands = avg_salary
    ),
  "score_lookup.csv"
)

cat("\n=== Lookup Table Summary ===\n")
cat("Total rows:", nrow(lookup_table), "\n")
cat("States:", n_distinct(lookup_table$state), "\n")
cat("Corporation types:", n_distinct(lookup_table$corp_type_code), "\n")
cat("Employee sizes:", n_distinct(lookup_table$emp_size_code), "\n")
cat("\nScore distribution:\n")
print(summary(lookup_table$predicted_score))

cat("\n=== Sample Output ===\n")
print(head(lookup_table %>% select(state, corp_type_code, emp_size_code, predicted_score, confidence), 20))

cat("\n✓ Lookup table saved to score_lookup.csv\n")
