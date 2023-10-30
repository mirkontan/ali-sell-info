import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import pandas as pd
import os
import datetime
from PIL import Image

# Define a function to generate a timestamp
def generate_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def remove_whitespace(df, columns):
    for column in columns:
        df[column] = df[column].str.replace(r'\s', '', regex=True)


# Set the page title
st.set_page_config(page_title='Sellers Business Licence OCR Reader', layout='wide')

# Create a Streamlit app
st.markdown("<h1 style='text-align: center;'>Sellers Business Licence OCR Reader</h1>", unsafe_allow_html=True)

# Upload Screenshots of Visure
st.sidebar.header('Upload screeshots of Sellers Info:')
uploaded_images = st.sidebar.file_uploader('Upload JPG or PNG images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
st.sidebar.subheader(f"{len(uploaded_images)} screenshot(s) have been uploaded.")

# Upload XLSX file with sellers records
uploaded_xlsx = st.sidebar.file_uploader('Upload XLSX file:', type=['xlsx'])

# Create a DataFrame to store the extracted text
df_sellers_info = pd.DataFrame()

# Check if the XLSX file has been uploaded
if uploaded_xlsx:
    # Load the XLSX file into a DataFrame
    df_urls = pd.read_excel(uploaded_xlsx)
    # Remove rows where 'SCREENSHOT' column is not empty
    df_urls = df_urls[df_urls['SCREENSHOT'].isnull()]
    # Reset the index of df_urls
    df_urls = df_urls.reset_index(drop=True)

    st.write(df_urls)
    st.sidebar.subheader(f"{len(df_urls['SELLER'])} seller(s) record(s) have been uploaded.")

if uploaded_images:
    # Initialize a dictionary to store the extracted text for each target
    # extracted_data = {}
    df_extraction = pd.DataFrame()
    df_extraction_aliexpress = pd.DataFrame()
    df_extraction_test = pd.DataFrame()

    for i, uploaded_image in enumerate(uploaded_images):
        # Display the uploaded image
        st.image(uploaded_image, use_column_width=True, caption='Uploaded Image')

        # Initialize a dictionary for this image
        extracted_data_per_image = {'SELLER': None, 'SELLER_URL': None}
        # Add the filename to the dictionary
        extracted_data_per_image['FILENAME'] = uploaded_image.name

        # Perform OCR on the entire uploaded image (CN language)
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        ocr_text = pytesseract.image_to_string(image)
       

        if not uploaded_xlsx:  
            # Check platform in the OCR text
            st.write(ocr_text)
            if 'Informazioni' in ocr_text:
                platform = 'ALIEXPRESS'
            else:
                platform = 'UNKNOWN'
           
        else:
            row = df_urls.iloc[i]  # We want to match with the first row of df_urls
            extracted_data_per_image['SELLER'] = row['SELLER']
            extracted_data_per_image['SELLER_URL'] = row['SELLER_URL']
            platform = row['PLATFORM']


        # Assign the platform to the dictionary for this image
        extracted_data_per_image['PLATFORM'] = platform
        
        # Create a selection box to allow the user to select the platform
        # Define the platform options
        platform_options = ['ALIEXPRESS', 'UNKNOWN']
        
        # Create a selection box with a unique key based on the filename
        selected_platform = st.selectbox(f'Select the platform for {uploaded_image.name}', platform_options, index=platform_options.index(extracted_data_per_image['PLATFORM']), key=f"selectbox_{i}")
        # Update the PLATFORM value with the selected platform
        extracted_data_per_image['PLATFORM'] = selected_platform
        
        # Display a small preview of the uploaded image
        st.image(uploaded_image, use_column_width=False, caption=f'Uploaded Image: {uploaded_image.name}', width=400)
        
        # Set the target words to look up in each image
        targets_aliexpress = ['卖家', '卖 家', '企业名称', '注册号', '注则号', '所在地', '地址', '网址', '法定代表人', '注册资本', '有效期', '经营范围', '经营学围', '店铺名称']
        targets_test = ['企业注册号', '注册号', '企业名称', '公司名称', '类 型', '类 ”型', '类 。 型', '类型', '地址', '住所', '住 所', '住 ”所']

        
        if extracted_data_per_image['PLATFORM'] == 'ALIEXPRESS':
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='ita', config='--psm 6')
            st.write('color image ocr text')
            st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            st.write(lines)
            data = {}
            current_key = None

            for line in lines:
                # Split based on a colon (":")
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    data[key] = value
                    current_key = key
                else:
                    # If no colon found, append to the previous key if available
                    if current_key is not None:
                        data[current_key] += ' ' + line.strip()            
            # Assign the 'PLATFORM' and 'FILENAME' values in the extracted_data_per_image dictionary
            extracted_data_per_image_aliexpress = pd.DataFrame([data])
            extracted_data_per_image_aliexpress['PLATFORM'] = 'ALIEXPRESS'
            extracted_data_per_image_aliexpress['FILENAME'] = uploaded_image.name
            st.write(extracted_data_per_image_aliexpress)
            st.write(data)
            df_extraction_aliexpress = pd.concat([df_extraction_aliexpress, extracted_data_per_image_aliexpress], ignore_index=True)

            targets = targets_aliexpress
        


        if extracted_data_per_image['PLATFORM'] == 'UNKNOWN' or None:
            targets = targets_test
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='ita', config='--psm 6')
            st.write('color image ocr text')
            st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            st.write(lines)
            data = {}
            current_key = None

            for line in lines:
                # Split based on a colon (":")
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    data[key] = value
                    current_key = key
                else:
                    # If no colon found, append to the previous key if available
                    if current_key is not None:
                        data[current_key] += ' ' + line.strip()            
            # Assign the 'PLATFORM' and 'FILENAME' values in the extracted_data_per_image dictionary
            extracted_data_per_image_test = pd.DataFrame([data])
            extracted_data_per_image_test['PLATFORM'] = 'UNKNOWN'
            extracted_data_per_image_test['FILENAME'] = uploaded_image.name
            st.write(extracted_data_per_image_test)
            st.write(data)
            df_extraction_test = pd.concat([df_extraction_test, extracted_data_per_image_test], ignore_index=True)

            
        # Display the entire extracted text
        st.subheader("Entire Extracted Text")
        st.write(ocr_text)


    df_extraction_overall = pd.concat([df_extraction_aliexpress, df_extraction_test], ignore_index=True)
    df_extraction_overall = pd.concat([df_extraction_overall, df_extraction_tmall], ignore_index=True)
    st.header('df EXTRACTION TEST')
    st.write(df_extraction_test)
    st.header('DF EXTRACTION ALIEXPRESS')
    st.write(df_extraction_aliexpress)

    st.header('DF EXTRACTION OVERALL')
    st.write(df_extraction_overall)

    # Copy the DataFrame
    df_sellers_info = df_extraction_overall.copy()

    
    aliexpress_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'ALIEXPRESS']
    test_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'UNKNOWN']
    


# ------------------------------------------------------------
#                             ALIEXPRESS
# ------------------------------------------------------------

    if aliexpress_df.empty:
        # Create new columns based on the items in targets_taobao
        # targets_aliexpress = ['Nome della società', 'Nome della socletà']
        for target in targets_aliexpress:
            jd_df[target] = None  # Add new columns with None values
    else:
        try:
            aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['Nome della società']
        except KeyError:
            aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['Company Name']

        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r'_.','')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r"'",'')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r"‘",'')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r"Co., Lt",'Co., Ltd')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r"Co., Lig",'Co., Ltd')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r"Co Lia",'Co., Ltd')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r"Co., Lîd",'Co., Ltd')
        # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.replace(r'Co., Ltd.*$', 'Co., Ltd', regex=True)
        # Remove leading and trailing spaces
        aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.strip()

        aliexpress_df['COMPANY_TYPE'] = '-'

        try:
            aliexpress_df['SELLER_VAT_N'] = aliexpress_df['Partita.IVA']
        except KeyError:
            aliexpress_df['SELLER_VAT_N'] = aliexpress_df['Partita IVA']
        # Remove 'Numero di' and the text before it if it's present
        aliexpress_df['SELLER_VAT_N'] = aliexpress_df['SELLER_VAT_N'].str.replace(r'Numero di.*$', '', regex=True)
        # aliexpress_df['SELLER_VAT_N'] = aliexpress_df['SELLER_VAT_N'].str.replace(r'registrazione.*$', '', regex=True)
        # Remove leading and trailing spaces
        aliexpress_df['SELLER_VAT_N'] = aliexpress_df['SELLER_VAT_N'].str.strip()

        try:
            aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['Stabilito']
        except KeyError:
            aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['. Stabilito']
        
        
        aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.strip()
        aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.replace('Autorità di', '-', regex=False)
        aliexpress_df['REGISTRATION_INSTITUTION'] = aliexpress_df['ESTABLISHED_IN'].str.split(' - ').str[1]
        aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.split(' - ').str[0]

        # aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.replace(r'..*$', '', regex=True)
        # aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.replace(r' .*$', '', regex=True)
        # aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.replace(r'.', '', regex=False)
     
        aliexpress_df['SELLER_ADDRESS'] = aliexpress_df['Indirizzo']

        try:
            aliexpress_df['SELLER_EMAIL'] = aliexpress_df['E-mail']
        except KeyError:
            aliexpress_df['SELLER_EMAIL'] = aliexpress_df['E-mall']

        aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.strip()
        aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.split(' ').str[0]

        # aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace(r'outlook con.*$', 'outlook.com', regex=True)
        # aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace(r'.com.*$', '.com', regex=True)
        # aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace(r'0163.*$', '@163.com', regex=True)
        # aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace(r'@ ', '@', regex=True)
        # aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace(r' outlook', '@outlook', regex=True)
        # aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace(r'00g.com', '@qq.com', regex=True)
  
        aliexpress_df['SELLER_TEL_N'] = aliexpress_df['Numero di telefono']

        aliexpress_df['LEGAL_REPRESENTATIVE'] = aliexpress_df['Rappresentante legale']

        aliexpress_df['BUSINESS_DESCRIPTION'] = aliexpress_df['Ambito di attività']


# ------------------------------------------------------------
#                             TEST - UNKNOWN PLATFORMS
# ------------------------------------------------------------

    if test_df.empty:
        # Create new columns based on the items in targets_taobao
        for target in targets_test:
            test_df[target] = None  # Add new columns with None values
    else:
        # # Check for missing values and replace them with an empty string
        test_df['SELLER_VAT_N'] = test_df['企业注册号'].fillna('') + test_df['注册号'].fillna('')
        test_df['SELLER_VAT_N'].fillna('', inplace=True)
        test_df['SELLER_VAT_N'] = test_df['SELLER_VAT_N'].str.split(':').str[1]
        test_df['SELLER_VAT_N'] = test_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
       
        test_df['SELLER_BUSINESS_NAME_CN'] = test_df['企业名称'].fillna('') + test_df['公司名称'].fillna('')
        
        test_df['COMPANY_TYPE_CN'] = test_df['类 型'].fillna('') + test_df['类 ”型'].fillna('') + test_df['类 。 型'].fillna('') + test_df['类型'].fillna('')
        test_df['COMPANY_TYPE_CN'] = test_df['COMPANY_TYPE'].str.split('住').str[0]
        test_df['COMPANY_TYPE_CN'] = test_df['COMPANY_TYPE'].str.split('地址').str[0]
        test_df['COMPANY_TYPE_CN'] = test_df['COMPANY_TYPE'].str.split('|').str[0]
        test_df['COMPANY_TYPE_CN'] = test_df['COMPANY_TYPE'].str.replace(r'型', '', regex=False)

        test_df['SELLER_ADDRESS_CN'] = test_df['住 所'].fillna('') + test_df['住 ”所'].fillna('') + test_df['住所'].fillna('') + test_df['地址'].fillna('')
        test_df['SELLER_ADDRESS_CN'] = test_df['SELLER_ADDRESS_CN'].str.split('法定').str[0]
        test_df['SELLER_ADDRESS_CN'] = test_df['SELLER_ADDRESS_CN'].str.split('|').str[0]
        test_df['SELLER_ADDRESS_CN'] = test_df['SELLER_ADDRESS_CN'].str.upper()

        test_df['LEGAL_REPRESENTATIVE_CN'] = test_df['法定代表人'].str.split('成').str[0]



        test_df['BUSINESS_DESCRIPTION'] = test_df['经营范围'].fillna('') + test_df['经营学围'].fillna('')

        test_df['ESTABLISHED_IN'] = test_df['成立时间'].str.split('注').str[0]
        test_df['ESTABLISHED_IN'] = test_df['ESTABLISHED_IN'].str.split('|').str[0]

        test_df['INITIAL_CAPITAL'] = test_df['注册资本'].str.split('营').str[0]
        test_df['INITIAL_CAPITAL'] = test_df['INITIAL_CAPITAL'].str.split('|').str[0]

        test_df['EXPIRATION_DATE'] = test_df['营业期限'].str.split('经').str[0]
        test_df['EXPIRATION_DATE'] = test_df['EXPIRATION_DATE'].str.split('|').str[0]

        test_df['REGISTRATION_INSTITUTION'] = test_df['登记机关'].str.split('核').str[0]
        test_df['REGISTRATION_INSTITUTION'] = test_df['REGISTRATION_INSTITUTION'].str.split('|').str[0]
        test_df['REGISTRATION_INSTITUTION'] = test_df['REGISTRATION_INSTITUTION'].str.split('注').str[0]
        test_df['SELLER_ADDRESS_CITY'] = '-'
        test_df['SHOP_NAMEextracted'] = '-'
        test_df['SHOP_URLextracted'] = '-'

    
    # Concatenate them into a single DataFrame
    sellers_info_df = pd.concat([aliexpress_df, test_df], ignore_index=True)
    
    st.header('SELLER INFO-1')
    st.write(sellers_info_df)
    
    # Count the number of rows in sellers_info_df
    num_rows = sellers_info_df.shape[0]

    # Display the number of rows
    st.sidebar.subheader(f"{num_rows} seller(s) have been analysed")

    country_city_dict = {
        ('Shenzhen', 'Guangdong', 'Mainland China'),
        ('Guangzhou', 'Guangdong', 'Mainland China'),
        ('Bengbu', 'Anhui', 'Mainland China'),
        ('Nanning', 'Guangxi', 'Mainland China'),
        ('Shanghai', 'Shanghai', 'Mainland China')
    }
    # Function to extract city, province, and country
    def extract_city_and_country(address):
        for city, province, country in country_city_dict:
            if city in address:
                return city, province, country
        return None, None, None

    # Apply the extraction function to each row
    sellers_info_df['SELLER_CITY_'], sellers_info_df['SELLER_PROVINCE_'], sellers_info_df['SELLER_COUNTRY'] = zip(*sellers_info_df['SELLER_ADDRESS'].apply(extract_city_and_country))
    # Update 'SELLER_PROVINCE' if 'Province Not Found' and 'SELLER_PROVINCE_' is not None
    sellers_info_df.loc[(sellers_info_df['SELLER_PROVINCE'] == 'Province Not Found') & (sellers_info_df['SELLER_PROVINCE_'].notna()), 'SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE_']

    # Update 'SELLER_CITY' if 'City Not Found' and 'SELLER_CITY_' is not None
    sellers_info_df.loc[(sellers_info_df['SELLER_CITY'] == 'City not found') & (sellers_info_df['SELLER_CITY_'].notna()), 'SELLER_CITY'] = sellers_info_df['SELLER_CITY_']
    sellers_info_df['SELLER_CITY'] = sellers_info_df['SELLER_CITY'].str.replace(' City', '', regex=False)
    sellers_info_df['SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE'].str.replace(' Province', '', regex=False)

    # Display the DataFrame with extracted text
    st.subheader("Extracted Text Data")
    st.write(sellers_info_df)


    # Add the download link
    download_link = st.button("Export to Excel (XLSX)")

    if download_link:
        # Generate a timestamp for the filename
        timestamp = generate_timestamp()
        filename = f"SellersInfo_{timestamp}.xlsx"
        
        # Define the path to save the Excel file
        download_path = os.path.join("/Users/mirkofontana/Downloads", filename)

        # Export the DataFrame to Excel
        sellers_info_df.to_excel(download_path, index=False)

        # Provide the download link
        st.markdown(f"Download the Excel file: [SellersInfo_{timestamp}.xlsx]({download_path})")






