data <- read.csv("census-income.data",sep=',',header=FALSE)
head(data)
names(data) <- c("age","type_employer","industry_code","occupation_code","education","wage","education_level",
                 "marital_status","industry","occupation","race","hispanic","sex","labor_union","unemp_reason",
                 "employment_status","cap_gain","cap_loss","dividends","tax_filer_status","region","state",
                 "household_details","household_summary","instance_weights","migr_msa","migr_inter_reg",
                 "migr_reg","same_home_yr","migr_from_sunbelt","num_working_for_empr","parents_present",
                 "father_bcountry","mother_bcountry","self_bcountry","citizenship","self_emp","vet_qn","vet_benefits","weeks_worked",
                 "year","income")

write.table(data,"train_raw.csv",sep=",",row.names=TRUE)
names(data)
## data transformations - aim reduce the categories

#age
table(data$age)

## employer type : converted Local and state govt employees to other govt as they exhibited 
## similar income patterns. Grouped self employed categories together
## grouped without-pay and not working together
## reduced to 6 from 9 categories

table(data$type_employer,data$income)
data$type_employer = as.character(data$type_employer)
data$type_employer = gsub("Federal government","Federal-Govt",data$type_employer)
data$type_employer = gsub("Local government","Other-Govt",data$type_employer)
data$type_employer = gsub("State government","Other-Govt",data$type_employer)
data$type_employer = gsub("Private","Private",data$type_employer)
data$type_employer = gsub("Self-employed-incorporated","Self-Employed",data$type_employer)
data$type_employer = gsub("Self-employed-not incorporated","Self-Employed",data$type_employer)
data$type_employer = gsub("Without pay","Not-Working",data$type_employer)
data$type_employer = gsub("Never worked","Not-Working",data$type_employer)
data$type_employer = gsub("Not in universe","Not-in-universe",data$type_employer)

## delete industry_code and occupation code since there is occupation and industry?
table(data$occupation,data$income)
table(data$industry,data$income)




#RACHAN There is education and education level. can get rid of one of these?!
#education
table(data$education,data$income)
data$education = as.character(data$education)
data$education = gsub("10th grade","Dropout",data$education)
data$education = gsub("11th grade","Dropout",data$education)
data$education = gsub("12th grade no diploma","Dropout",data$education)
data$education = gsub("1st 2nd 3rd or 4th grade","Dropout",data$education)
data$education = gsub("5th or 6th grade","Dropout",data$education)
data$education = gsub("7th and 8th grade","Dropout",data$education)
data$education = gsub("9th grade","Dropout",data$education)
data$education = gsub("Associates degree-academic program","Associates",data$education)
data$education = gsub("Associates degree-occup /vocational","Associates",data$education)
data$education = gsub("Bachelors.*","Bachelors",data$education)
data$education = gsub("Doctorate.*","Doctorate",data$education)
data$education = gsub("High school graduate","HS-Graduate",data$education)
data$education = gsub("Masters.*","Masters",data$education)
#data$education = gsub("^Preschool","Dropout",data$education)
data$education = gsub("Prof school.*","Prof-Degree",data$education)
data$education = gsub("Some college but no degree","HS-Graduate",data$education)

#wage
table(data$wage)

#education_level
table(data$education_level,data$income)

#marital_status
table(data$marital_status,data$income)

#data$education = as.character(data$education)

data$marital_status = gsub("Never married","not-married",data$marital_status)
data$marital_status = gsub("Married-A F spouse present","married",data$marital_status)
data$marital_status = gsub("Married-civilian spouse present","married",data$marital_status)
data$marital_status = gsub("Married-spouse absent","married",data$marital_status)


#industry-do we have to aggr?
table(data$industry,data$income)

#occupation
table(data$occupation,data$income)

#race
table(data$race)


#hispanic
table(data$hispanic)
data$hispanic = as.character(data$hispanic)
ux <- unique(data$hispanic) 
common <-ux[which.max(tabulate(match(data$hispanic, ux)))]
data$hispanic[which(data$hispanic==NA)] = common

#unemployement reason
table(data$unemp_reason,data$income)
data$unemp_reason = as.character(data$unemp_reason)
data$unemp_reason = gsub("Job leaver","job-leaver",data$unemp_reason)
data$unemp_reason = gsub("Job loser.*","job-loser",data$unemp_reason)
data$unemp_reason = gsub("Other job loser.*","job-loser",data$unemp_reason)
data$unemp_reason = gsub("New entrant","new-entrant",data$unemp_reason)
data$unemp_reason = gsub("Re-entrant","re-entrant",data$unemp_reason)

#employment_status

table(data$employment_status,data$income)
data$employment_status = as.character(data$employment_status)
data$employment_status = gsub("Full-time schedules","Full-time",data$employment_status)
data$employment_status = gsub("PT for econ reasons usually FT","Part-time",data$employment_status)
data$employment_status = gsub("PT for econ reasons usually PT","Part-time",data$employment_status)
data$employment_status = gsub("PT for non-econ reasons usually FT","Part-time",data$employment_status)
data$employment_status = gsub("Unemployed full-time","Unemployed",data$employment_status)
data$employment_status = gsub("Unemployed part- time","Unemployed",data$employment_status)

#tax_filer_status-check if aggr is required
table(data$tax_filer_status,data$income)

#region
table(data$region)

#state there are missing values
table(data$state, data$income)
data$state = as.character(data$state)
names(which.max(table(data$state)))
data$state[data$state==" ?"] <- " Not in universe"


#household_details - aggregate 
#RACHAN very similar to household_summary?????
table(data$household_details, data$income)




#household_summary
table(data$household_summary,data$income)
data$household_summary = as.character(data$household_summary)
data$household_summary = gsub("Child 18 or older","child-above-18",data$household_summary)
data$household_summary = gsub("Child under 18 ever married","child-below-18",data$household_summary)
data$household_summary = gsub("Child under 18 never married","child-below-18",data$household_summary)
data$household_summary = gsub("Group Quarters- Secondary individual","nonrelative",data$household_summary)
data$household_summary = gsub("Nonrelative of householder","nonrelative",data$household_summary)
data$household_summary = gsub("Other relative of householder","relative",data$household_summary)
data$household_summary = gsub("Spouse of householder","spouse",data$household_summary)

#migration to metropolitan areas- should we aggregate?-missing vals
#RACHAN not sure how to aggregate further?
table(data$migr_msa,data$income)
data$migr_msa = as.character(data$migr_msa)
ux <- unique(data$migr_msa) 
common <-ux[which.max(tabulate(match(data$migr_msa, ux)))]
data$migr_msa[data$migr_msa==" ?"] <- " Nonmover"

#migration within the same region-missing vals
table(data$migr_inter_reg,data$income)
data$migr_inter_reg = as.character(data$migr_inter_reg)
data$migr_inter_reg[data$migr_inter_reg==" ?"] <- " Nonmover"

#migration between regions-missing vals
table(data$migr_reg,data$income)
data$migr_reg = as.character(data$migr_reg)
data$migr_reg[data$migr_reg==" ?"] <- " Nonmover"

#lived in the same home the last one yr- No need to aggregate
table(data$same_home_yr,data$income) 

#migration from sunbelt-do we have to include?-missing vals
table(data$migr_from_sunbelt,data$income)
data$migr_from_sunbelt = as.character(data$migr_from_sunbelt)
ux <- unique(data$migr_from_sunbelt) 
common <-ux[which.max(tabulate(match(data$migr_from_sunbelt, ux)))]
data$migr_from_sunbelt[data$migr_from_sunbelt==" ?"] <- " Not in universe"

#parents present- do we include?
table(data$parents_present,data$income)


#birth country - father-if including aggr into regions - look up http://scg.sdsu.edu/dataset-adult_r/
#RACHAN - aggregated according to the groups on above website
table(data$father_bcountry,data$income)
data$father_bcountry = as.character(data$father_bcountry)
data$father_bcountry = gsub("Cambodia", "SE-Asia", data$father_bcountry)
data$father_bcountry = gsub("Canada" , "British-Commonwealth", data$father_bcountry)    
data$father_bcountry = gsub("China" ,"China", data$father_bcountry)       
data$father_bcountry = gsub("Columbia"  , "South-America"    , data$father_bcountry)
data$father_bcountry = gsub("Cuba"  , "Other", data$father_bcountry)        
data$father_bcountry = gsub("Dominican-Republic"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Ecuador"  , "South-America"     , data$father_bcountry)
data$father_bcountry = gsub("El-Salvador"  , "South-America" , data$father_bcountry)
data$father_bcountry = gsub("England"  , "British-Commonwealth", data$father_bcountry)
data$father_bcountry = gsub("France"  , "Euro_1", data$father_bcountry)
data$father_bcountry = gsub("Germany"  , "Euro_1", data$father_bcountry)
data$father_bcountry = gsub("Greece"  , "Euro_2", data$father_bcountry)
data$father_bcountry = gsub("Guatemala"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Haiti"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Holand-Netherlands"  , "Euro_1", data$father_bcountry)
data$father_bcountry = gsub("Honduras"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Hong Kong"  , "China", data$father_bcountry)
data$father_bcountry = gsub("Hungary"  , "Euro_2", data$father_bcountry)
data$father_bcountry = gsub("India"  , "British-Commonwealth", data$father_bcountry)
data$father_bcountry = gsub("Ireland"  , "British-Commonwealth", data$father_bcountry)
data$father_bcountry = gsub("Italy"  , "Euro_1", data$father_bcountry)
data$father_bcountry = gsub("Jamaica"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Japan"  , "Other", data$father_bcountry)
data$father_bcountry = gsub("Laos"  , "SE-Asia", data$father_bcountry)
data$father_bcountry = gsub("Mexico"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Nicaragua"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Outlying-U S.*"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Panama", "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Peru"  , "South-America", data$father_bcountry)
data$father_bcountry = gsub("Philippines"  , "SE-Asia", data$father_bcountry)
data$father_bcountry = gsub("Poland"  , "Euro_2", data$father_bcountry)
data$father_bcountry = gsub("Portugal"  , "Euro_2", data$father_bcountry)
data$father_bcountry = gsub("Puerto-Rico"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("Scotland"  , "British-Commonwealth", data$father_bcountry)
data$father_bcountry = gsub("South Korea"  , "SE-Asia", data$father_bcountry)
data$father_bcountry = gsub("Taiwan"  , "China", data$father_bcountry)
data$father_bcountry = gsub("Thailand"  , "SE-Asia", data$father_bcountry)
data$father_bcountry = gsub("Trinadad&Tobago"  , "Latin-America", data$father_bcountry)
data$father_bcountry = gsub("United-States"  , "United-States", data$father_bcountry)
data$father_bcountry = gsub("Vietnam"  , "SE-Asia", data$father_bcountry)
data$father_bcountry = gsub("Yugoslavia"  , "Euro_2", data$father_bcountry)
ux <- unique(data$father_bcountry) 
common <-ux[which.max(tabulate(match(data$father_bcountry , ux)))]
data$father_bcountry [data$father_bcountry==" ?"] <- common

#birth country - mother-if including aggr into regions - look up http://scg.sdsu.edu/dataset-adult_r/
#RACHAN - aggregated according to the groups on above websites
table(data$mother_bcountry,data$income)
data$mother_bcountry = as.character(data$mother_bcountry)
data$mother_bcountry = gsub("Cambodia", "SE-Asia", data$ mother_bcountry)
data$mother_bcountry = gsub("Canada" , "British-Commonwealth", data$mother_bcountry)    
data$mother_bcountry = gsub("China" ,"China", data$mother_bcountry)       
data$mother_bcountry = gsub("Columbia"  , "South-America"    , data$mother_bcountry)
data$mother_bcountry = gsub("Cuba"  , "Other", data$mother_bcountry)        
data$mother_bcountry = gsub("Dominican-Republic"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Ecuador"  , "South-America"     , data$mother_bcountry)
data$mother_bcountry = gsub("El-Salvador"  , "South-America" , data$mother_bcountry)
data$mother_bcountry = gsub("England"  , "British-Commonwealth", data$mother_bcountry)
data$mother_bcountry = gsub("France"  , "Euro_1", data$mother_bcountry)
data$mother_bcountry = gsub("Germany"  , "Euro_1", data$mother_bcountry)
data$mother_bcountry = gsub("Greece"  , "Euro_2", data$mother_bcountry)
data$mother_bcountry = gsub("Guatemala"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Haiti"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Holand-Netherlands"  , "Euro_1", data$mother_bcountry)
data$mother_bcountry = gsub("Honduras"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Hong Kong"  , "China", data$mother_bcountry)
data$mother_bcountry = gsub("Hungary"  , "Euro_2", data$mother_bcountry)
data$mother_bcountry = gsub("India"  , "British-Commonwealth", data$mother_bcountry)
data$mother_bcountry = gsub("Ireland"  , "British-Commonwealth", data$mother_bcountry)
data$mother_bcountry = gsub("Italy"  , "Euro_1", data$mother_bcountry)
data$mother_bcountry = gsub("Jamaica"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Japan"  , "Other", data$mother_bcountry)
data$mother_bcountry = gsub("Laos"  , "SE-Asia", data$mother_bcountry)
data$mother_bcountry = gsub("Mexico"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Nicaragua"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Outlying-U S.*"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Panama", "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Peru"  , "South-America", data$mother_bcountry)
data$mother_bcountry = gsub("Philippines"  , "SE-Asia", data$mother_bcountry)
data$mother_bcountry = gsub("Poland"  , "Euro_2", data$mother_bcountry)
data$mother_bcountry = gsub("Portugal"  , "Euro_2", data$mother_bcountry)
data$mother_bcountry = gsub("Puerto-Rico"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("Scotland"  , "British-Commonwealth", data$mother_bcountry)
data$mother_bcountry = gsub("South Korea"  , "SE-Asia", data$mother_bcountry)
data$mother_bcountry = gsub("Taiwan"  , "China", data$mother_bcountry)
data$mother_bcountry = gsub("Thailand"  , "SE-Asia", data$mother_bcountry)
data$mother_bcountry = gsub("Trinadad&Tobago"  , "Latin-America", data$mother_bcountry)
data$mother_bcountry = gsub("United-States"  , "United-States", data$mother_bcountry)
data$mother_bcountry = gsub("Vietnam"  , "SE-Asia", data$mother_bcountry)
data$mother_bcountry = gsub("Yugoslavia"  , "Euro_2", data$mother_bcountry)
ux <- unique(data$mother_bcountry) 
common <-ux[which.max(tabulate(match(data$mother_bcountry, ux)))]
data$mother_bcountry [data$mother_bcountry==" ?"] <- common


#self birth country -aggr- look up http://scg.sdsu.edu/dataset-adult_r/ to see how to aggregate
#RACHAN aggregated accordcing to groups on above website
table(data$self_bcountry,data$income)
data$self_bcountry = as.character(data$self_bcountry)
data$self_bcountry = gsub("Cambodia", "SE-Asia", data$self_bcountry)
data$self_bcountry = gsub("Canada" , "British-Commonwealth", data$self_bcountry)    
data$self_bcountry = gsub("China" ,"China", data$self_bcountry)       
data$self_bcountry = gsub("Columbia"  , "South-America"    , data$self_bcountry)
data$self_bcountry = gsub("Cuba"  , "Other", data$self_bcountry)        
data$self_bcountry = gsub("Dominican-Republic"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Ecuador"  , "South-America"     , data$self_bcountry)
data$self_bcountry = gsub("El-Salvador"  , "South-America" , data$self_bcountry)
data$self_bcountry = gsub("England"  , "British-Commonwealth", data$self_bcountry)
data$self_bcountry = gsub("France"  , "Euro_1", data$self_bcountry)
data$self_bcountry = gsub("Germany"  , "Euro_1", data$self_bcountry)
data$self_bcountry = gsub("Greece"  , "Euro_2", data$self_bcountry)
data$self_bcountry = gsub("Guatemala"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Haiti"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Holand-Netherlands"  , "Euro_1", data$self_bcountry)
data$self_bcountry = gsub("Honduras"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Hong Kong"  , "China", data$self_bcountry)
data$self_bcountry = gsub("Hungary"  , "Euro_2", data$self_bcountry)
data$self_bcountry = gsub("India"  , "British-Commonwealth", data$self_bcountry)
data$self_bcountry = gsub("Ireland"  , "British-Commonwealth", data$self_bcountry)
data$self_bcountry = gsub("Italy"  , "Euro_1", data$self_bcountry)
data$self_bcountry = gsub("Jamaica"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Japan"  , "Other", data$self_bcountry)
data$self_bcountry = gsub("Laos"  , "SE-Asia", data$self_bcountry)
data$self_bcountry = gsub("Mexico"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Nicaragua"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Outlying-U S.*"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Panama", "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Peru"  , "South-America", data$self_bcountry)
data$self_bcountry = gsub("Philippines"  , "SE-Asia", data$self_bcountry)
data$self_bcountry = gsub("Poland"  , "Euro_2", data$self_bcountry)
data$self_bcountry = gsub("Portugal"  , "Euro_2", data$self_bcountry)
data$self_bcountry = gsub("Puerto-Rico"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("Scotland"  , "British-Commonwealth", data$self_bcountry)
data$self_bcountry = gsub("South Korea"  , "SE-Asia", data$self_bcountry)
data$self_bcountry = gsub("Taiwan"  , "China", data$self_bcountry)
data$self_bcountry = gsub("Thailand"  , "SE-Asia", data$self_bcountry)
data$self_bcountry = gsub("Trinadad&Tobago"  , "Latin-America", data$self_bcountry)
data$self_bcountry = gsub("United-States"  , "United-States", data$self_bcountry)
data$self_bcountry = gsub("Vietnam"  , "SE-Asia", data$self_bcountry)
data$self_bcountry = gsub("Yugoslavia"  , "Euro_2", data$self_bcountry)
ux <- unique(data$self_bcountry) 
common <-ux[which.max(tabulate(match(data$self_bcountry, ux)))]
data$self_bcountry [data$self_bcountry==" ?"] <- common


#citizenship
table(data$citizenship,data$income)
data$citizenship = as.character(data$citizenship)
data$citizenship = gsub("Foreign born.*","foreigner",data$citizenship)
data$citizenship = gsub("Foreign born.*","foreigner",data$citizenship)
data$citizenship = gsub("Native- Born abroad of American Parent.*","american",data$citizenship)
data$citizenship = gsub("Native- Born in the United States","american",data$citizenship)
data$citizenship = gsub("Native- Born in Puerto Rico or U S Outlying","american-minor",data$citizenship)

#self-emp - own business or self employed
#0- not in universe 1 - yes 2 - no
table(data$self_emp,data$income)

#vet_qn , fill inc questionnaire for veteran's admin? should we include? yes/no are very less
#0- not in universe 1 - yes 2 - no
#RACHAN coded 0,1,2
table(data$vet_qn,data$income)
data$vet_qn = as.character(data$vet_qn)
data$vet_qn = gsub("Not in universe", "0", data$vet_qn)
data$vet_qn = gsub("Yes", "1", data$vet_qn)
data$vet_qn = gsub("No", "2", data$vet_qn)




#vet_benefits
#0- not in universe 1 - yes 2 - no
table(data$vet_benefits,data$income)

#year- should we include? distribution is pretty much the same
# 94 and 95
table(data$year,data$income)

#income- categorize it as 0's and 1's
table(data$income)
data$income = as.character(data$income)
data$income = gsub("- 50000.", "0", data$income)
data$income = gsub("+ 50000.", "1", data$income)



#summary(data==" ?")

write.table(data, file ="train_modified.csv",col.names=TRUE,row.names=FALSE,sep=",")

