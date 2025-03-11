const fs = require('fs');

class Data_Processing {

    constructor() {
        this.raw_user_data = []; 
        this.formatted_user_data = [];
        this.cleaned_user_data = [];
    }

    load_CSV(filename) {
        // reset the raw_user_data
        this.raw_user_data = [];

        // read the file,use readFileSync to block the execution until the file is read
        const fileContent = fs.readFileSync(`${filename}.csv`, 'utf8' );
        this.raw_user_data = fileContent;


        return this.raw_user_data;
    }

    
    format_data() {
        const rows = this.raw_user_data.split('\n').map(row => row.trim()).filter(row => row);//split by new line,trim the white space,filter out empty rows
    
        // Maps for number word conversion
        const ones = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
        };

        const tens = {
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        };

        const months = {
            'Jan': '01',
            'Feb': '02',
            'Mar': '03',
            'Apr': '04',
            'May': '05',
            'Jun': '06',
            'Jul': '07',
            'Aug': '08',
            'Sep': '09',
            'Oct': '10',
            'Nov': '11',
            'Dec': '12'
        };
    
        
    
        this.formatted_user_data = rows.map((row) => {//map each row to a formatted object
            const columns = row.split(',');
    
            // Assuming the columns are [fullName, dateOfBirth, age, email] as described
            const [fullName, dobRaw, ageRaw, email] = columns.map(col => col.trim());
    
            // Convert the age from string to integer
            let numericAge;

            if (isNaN(parseInt(ageRaw))) {
                let lowerCaseAge = ageRaw.toLowerCase();
                if (lowerCaseAge.includes("-")) {
                    // Handle hyphenated numbers like "twenty-one"
                    let ageParts = lowerCaseAge.split('-');
                    numericAge = (tens[ageParts[0]] || 0) + (ones[ageParts[1]] || 0);
                } else {
                    // Handle single-word numbers (both tens and teens)
                    numericAge = tens[lowerCaseAge] || ones[lowerCaseAge] || 0;
                }
            } else {
                numericAge = parseInt(ageRaw, 10);
            }
    
            // Handle date of birth formatting
            let dateOfBirth = '';
            let dobParts = dobRaw.split(/\/|\s/);
            if (dobParts.length === 3) {
                if (isNaN(dobParts[1])) {
                    dobParts[1] = months[dobParts[1].substring(0, 3)];
                }
                dobParts[2] = dobParts[2].length === 2 ? (parseInt(dobParts[2], 10) > 24 ? `19${dobParts[2]}` : `20${dobParts[2]}`) : dobParts[2];
                dobParts[0] = dobParts[0].padStart(2, '0');
                dobParts[1] = dobParts[1].padStart(2, '0');
            }
            dateOfBirth = `${dobParts[0]}/${dobParts[1]}/${dobParts[2]}`
    
            // Split fullName into title, first name, middle name, and surname
            let names = fullName.split(/\s+/);
            let title = names[0].match(/Mr|Mrs|Miss|Ms|Dr/) ? names.shift() : '';
            let firstName = names.shift() || '';
            let middleName = names.length > 1 ? names.slice(0, -1).join(' ') : '';
            let surname = names.pop() || '';
    
            // Format names
            firstName = firstName.charAt(0).toUpperCase() + firstName.slice(1).toLowerCase();
            middleName = middleName && middleName.charAt(0).toUpperCase() + middleName.slice(1).toLowerCase();
            surname = surname.charAt(0).toUpperCase() + surname.slice(1);
    
    
            return {
                title,
                first_name: firstName,
                middle_name: middleName,
                surname,
                date_of_birth: dateOfBirth,
                age: numericAge,// age is now an integer
                email: email
            };
        });
    
        
    }

    clean_data() {
        // Remove duplicate rows
        this.cleaned_user_data = JSON.parse(JSON.stringify(this.formatted_user_data.filter((user, index, self) =>
        index === self.findIndex((t) => (
            t.first_name === user.first_name && t.surname === user.surname && t.date_of_birth === user.date_of_birth
        )))));//copy the array to avoid reference,use deep copy!!!

        const collectionDate = new Date(2024, 1, 26); // JavaScript months are 0-indexed
        const emailBaseMap = {}; // used to track base email names and their occurrences

        this.cleaned_user_data.forEach((user, index) => {
            // Correct title "Dr." to "Dr"
            if (user.title.includes('.')) {
                //replace'.' with ''
                user.title = user.title.replace('.', '');
            }

            // calculate age according to date of birth
            if (user.date_of_birth) {
                const [day, month, year] = user.date_of_birth.split('/');
                const dob = new Date(parseInt(year, 10), parseInt(month, 10) - 1, parseInt(day, 10));
                let age = collectionDate.getFullYear() - dob.getFullYear();
                const m = collectionDate.getMonth() - dob.getMonth();
                if (m < 0 || (m === 0 && collectionDate.getDate() < dob.getDate())) {
                    age--;
                }
                user.age = age; // update age
            }

            // if first name or surname is missing, extract from email
            if (!user.first_name || !user.surname) {
                // assume email is in the format
                const nameParts = user.email.split('@')[0].split('.');
                if (nameParts.length >= 2) {
                    if (!user.first_name) {
                        user.first_name = nameParts[0];
                        // capitalize the first letter
                        user.first_name = user.first_name.charAt(0).toUpperCase() + user.first_name.slice(1).toLowerCase();
                    }
                    if (!user.surname) {
                        user.surname = nameParts[1];
                        // capitalize the first letter
                        user.surname = user.surname.charAt(0).toUpperCase() + user.surname.slice(1).toLowerCase();
                    }
                }
            }

            let baseEmail = `${user.first_name}.${user.surname}@example.com`;
            if (!emailBaseMap.hasOwnProperty(baseEmail)) {
                // if the base email name has not been seen before, add it to the map
                emailBaseMap[baseEmail] = [index];
            } else {
            // if the base email name has been seen before, add a number suffix
            const firstIndex = emailBaseMap[baseEmail][0];
            const occurrence = emailBaseMap[baseEmail].length + 1; // count the number of occurrences
            emailBaseMap[baseEmail].push(index);

            // update the first user's email
            const firstBaseEmail = `${baseEmail.split('@')[0]}1@example.com`;
            this.cleaned_user_data[firstIndex].email = firstBaseEmail;

            // add the number suffix to the current user's email
            baseEmail = `${baseEmail.split('@')[0]}${occurrence}@example.com`;
            user.email = baseEmail;
            }
            //update other user's email
            user.email = baseEmail;

        });
    }

    most_common_surname() {
        const surnameFrequency = {}; // used to track the frequency of each surname
        this.cleaned_user_data.forEach(user => {
            const { surname } = user;
            if (surnameFrequency[surname]) {
                surnameFrequency[surname] += 1; // if the surname is already in the map, increment the frequency
            } else {
                surnameFrequency[surname] = 1; // if the surname is not in the map, add it with a frequency of 1
            }
        });
    
        // find the highest frequency
        const maxFrequency = Math.max(...Object.values(surnameFrequency));
        // find all surnames with the highest frequency
        const mostCommonSurnames = Object.keys(surnameFrequency).filter(surname => surnameFrequency[surname] === maxFrequency);
    
        return mostCommonSurnames;
    }
    average_age() {
        if (this.cleaned_user_data.length === 0) {
            return 0; // if there are no users, return 0
        }
    
        // add up all the ages
        const totalAge = this.cleaned_user_data.reduce((sum, user) => sum + user.age, 0);
    
        // calculate the average
        const average = totalAge / this.cleaned_user_data.length;
    
        // fix the average to one decimal place
        return parseFloat(average.toFixed(1));
    }
    youngest_dr() {
        const doctors = this.cleaned_user_data.filter(user => user.title === 'Dr'); // find all doctors
    
        if (doctors.length === 0) return null; // if there are no doctors, return null
    
        // find the youngest doctor
        const youngestDoctor = doctors.reduce((youngest, current) => {
            return current.age < youngest.age ? current : youngest;
        }, doctors[0]); // initialize with the first doctor
    
        return youngestDoctor; // return the youngest doctor
    }
    most_common_month() {
        const monthFrequency = {}; // used to track the frequency of each month
    
        this.cleaned_user_data.forEach(user => {
            // assume date_of_birth is in the format 'dd/mm/yyyy'
            const month = parseInt(user.date_of_birth.split('/')[1], 10); // convert the month to an integer
            monthFrequency[month] = (monthFrequency[month] || 0) + 1;
        });
    
        // find the highest frequency
        const maxFrequency = Math.max(...Object.values(monthFrequency));
    
        // find all months with the highest frequency
        const mostCommonMonths = Object.keys(monthFrequency).filter(month => monthFrequency[month] === maxFrequency);
    
        // return the most common months as integers
        return mostCommonMonths.map(month => parseInt(month, 10));
    }
    
    percentage_titles() {
        const titleCounts = {
            'Mr': 0,
            'Mrs': 0,
            'Miss': 0,
            'Ms': 0,
            'Dr': 0,
            '': 0 // used for unknown or missing titles
        };
    
        // calculate the number of each title
        this.cleaned_user_data.forEach(user => {
            if (titleCounts.hasOwnProperty(user.title)) {
                titleCounts[user.title] += 1;
            } else {
                titleCounts[''] += 1; // for unknown or missing titles, increment the empty string key
            }
        });
    
        const totalUsers = this.cleaned_user_data.length;
        const percentages = [];
    
        // calculate the percentage of each title
        for (const title of ['Mr', 'Mrs', 'Miss', 'Ms', 'Dr', '']) {
            const percentage = Math.round((titleCounts[title] / totalUsers) * 100);
            percentages.push(percentage);
        }
    
        return percentages;
    }
    
    percentage_altered() {
        // compare the formatted data with the cleaned data
        let totalChanges = 0;
        
        const fieldsToCompare = ["title", "first_name", "middle_name", "surname", "date_of_birth", "age", "email"];//fields to compare
        for (let i = 0; i < this.cleaned_user_data.length; i++) {//compare each field of each user
          const formatted = this.formatted_user_data[i];
          const cleaned = this.cleaned_user_data[i];
          fieldsToCompare.forEach(field => {
            if (formatted[field] !== cleaned[field]) {//if the field is different, increment the total changes
              totalChanges++;
            }
          });
        }
        totalChanges += (this.formatted_user_data.length - this.cleaned_user_data.length) * fieldsToCompare.length;//add the difference in the number of users
    
        const percentage = (totalChanges / (this.formatted_user_data.length * fieldsToCompare.length)) * 100;//calculate the percentage of changes
    
        const resultWithThreeSignificantFigures = Number(percentage).toPrecision(3);
    
        return resultWithThreeSignificantFigures;
      }
}
    
    
    
    
    


