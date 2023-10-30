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
    df_extraction_tmall = pd.DataFrame()

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
        platform_options = ['ALIEXPRESS', 'TEST']
        
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
        


        if extracted_data_per_image['PLATFORM'] == 'TEST' or None:
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
            extracted_data_per_image_test['PLATFORM'] = 'TEST'
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
    test_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'TEST']
    


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
    sellers_info_df = pd.concat([tmall_df, taobao_df, chinaalibaba_df, jd_df, aliexpress_df, test_df], ignore_index=True)
   
   
   
    st.header('SELLER INFO-1')
    st.write(sellers_info_df)



    # Define the additional columns and their initial values
    # additional_columns = {
    #     "SELLER_BUSINESS_NAME_CN": '',
    #     'SELLER_BUSINESS_NAME': '',
    #     "COMPANY_TYPE_EN": '',
    #     "SELLER_ADDRESS_CN": '',
    #     "SELLER_PROVINCE_CN": '',
    #     "SELLER_CITY_CN": '',
    #     "LEGAL_REPRESENTATIVE_EN": '',
    #     'SELLER_ADDRESS': '',
    #     'SELLER_EMAIL': '',
    #     'SELLER_TEL_N': ''}
    # # Add the additional columns to the DataFrame
    # for column_name, initial_value in additional_columns.items():
    #     sellers_info_df[column_name] = initial_value



    sellers_info_df['SHOP_NAMEextracted'] = sellers_info_df['SHOP_NAMEextracted'].fillna('-')
    sellers_info_df['SHOP_URLextracted'] = sellers_info_df['SHOP_URLextracted'].fillna('-')
    sellers_info_df.drop(['统一社会', '企业注册号', '注册号', '公司名称', '企业名称', '企业类型', '地址', '成立日期', '注册号', '类型', '类 型', '类 ”型', '类 。 型', '地址', '住所', '住 所', '住 ”所', '法定代表人', '经营期限自', '经营范围', '经营学围', '成立时间', '注册资本', '营业期限', '登记机关'], axis=1, inplace=True)
    sellers_info_df['AIQICHA_URL'] = 'https://www.aiqicha.com/s?q=' + sellers_info_df['SELLER_VAT_N']
    sellers_info_df.drop(['该准时间'], axis=1, inplace=True)
    # Apply the translation function to the 'SELLER_BUSINESS_NAME_CN' column
    # sellers_info_df['SELLER_BUSINESS_NAME'] = sellers_info_df['SELLER_BUSINESS_NAME'].astype(str)
    # # Apply these transformations based on the 'PLATFORM' value
    # is_aliexpress = sellers_info_df['PLATFORM'] == 'ALIEXPRESS'
    # Apply transformations for 'ALIEXPRESS' platform
    # sellers_info_df.loc[is_aliexpress, 'SELLER_BUSINESS_NAME'] = sellers_info_df.loc[is_aliexpress, 'SELLER_BUSINESS_NAME']
    # Apply transformations for other platforms
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'].str.replace(r'贸易批发商行', 'Wholesale Trading Company', regex=True)
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'].str.replace(r'商行', 'Trading Company', regex=True)
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'].apply(translate_to_english)
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME'].str.replace(r'Trade Trading Company', 'Trading Company', regex=True)
    # sellers_info_df.loc[~is_aliexpress, 'COMPANY_TYPE_EN'] = sellers_info_df.loc[~is_aliexpress, 'COMPANY_TYPE'].apply(translate_to_english)
    # sellers_info_df.loc[~is_aliexpress, 'LEGAL_REPRESENTATIVE_EN'] = sellers_info_df.loc[~is_aliexpress, 'LEGAL_REPRESENTATIVE'].apply(translate_to_english)
    import googletrans
    from googletrans import Translator


    # Create a Translator instance
    translator = Translator()

    def fill_empty_with_translation(df, target_column, source_column):
        for index, row in df.iterrows():
            if pd.isna(row[target_column]) and not pd.isna(row[source_column]):
                try:
                    translation = translator.translate(row[source_column], src='zh-cn', dest='en')
                    df.at[index, target_column] = translation.text
                except Exception as e:
                    print(f"Translation error: {e}")

    fill_empty_with_translation(sellers_info_df, 'SELLER_BUSINESS_NAME', 'SELLER_BUSINESS_NAME_CN')
    fill_empty_with_translation(sellers_info_df, 'SELLER_ADDRESS', 'SELLER_ADDRESS_CN')
    fill_empty_with_translation(sellers_info_df, 'COMPANY_TYPE', 'COMPANY_TYPE_CN')
    fill_empty_with_translation(sellers_info_df, 'LEGAL_REPRESENTATIVE', 'LEGAL_REPRESENTATIVE_CN')

    st.header('SELLERS INFO TRANSLATED')
    st.write(sellers_info_df)


    import jieba

    # Sample Chinese addresses
    addresses = sellers_info_df['SELLER_ADDRESS_CN']
    # Function to extract city from an address using Jieba
    def extract_city(address):
        if isinstance(address, str):  # Check if it's a string
            words = list(jieba.cut(address, cut_all=False))  # Segment the address into words
            for word in words:
                if word.endswith('市'):
                    return word
                elif word.endswith('州'):   
                    return word + '市'
                elif word.endswith('圳'):
                    return word + '市'
        return "City not found" # Return a default value if the city is not in the address
    # Process each address
    # Create an empty 'SELLER_CITY' column
    sellers_info_df['SELLER_CITY_CN'] = ""

    # Process each address and assign the extracted city to the 'SELLER_CITY' column
    for index, address in enumerate(addresses):
        city = extract_city(address)
        sellers_info_df.at[index, 'SELLER_CITY_CN'] = city
    
    # Sample dictionary of Chinese cities to provinces
    city_to_province = {
        "北京市": "北京市",
        "上海市": "上海市",
        "广州市": "广东省",
        "深圳市": "广东省",
        "杭州市": "浙江省",
        "义乌市": "浙江省",
        "南京市": "江苏省",
        "成都市": "四川省",
        "重庆市": "重庆市",
        "武汉市": "湖北省",
        "西安市": "陕西省",
        "张家港市": "江苏省"
    }

    # Function to extract province from the city
    def extract_province(city):
        if city in city_to_province:
            return city_to_province[city]
        return "Province not found"  # Default value if the city is not in the mapping

    # Apply the extract_province function to the 'SELLER_CITY' column
    sellers_info_df['SELLER_PROVINCE_CN'] = sellers_info_df['SELLER_CITY_CN'].apply(extract_province)
   
    # Update 'SELLER_PROVINCE' and 'SELLER_CITY' if 'PLATFORM' is 'JD COM'
    condition = sellers_info_df['PLATFORM'] == 'JD COM'
    sellers_info_df.loc[condition & (sellers_info_df['SELLER_PROVINCE_CN'] == 'Province not found'), 'SELLER_PROVINCE_CN'] = sellers_info_df['SELLER_ADDRESS_CITY'].str.split('/').str[0]
    sellers_info_df.loc[condition & (sellers_info_df['SELLER_CITY_CN'] == 'City not found'), 'SELLER_CITY_CN'] = sellers_info_df['SELLER_ADDRESS_CITY'].str.split('/').str[1]
    
    st.write(sellers_info_df)
    # Apply the translation function to the 'SELLER_ADDRESS' column
    sellers_info_df['SELLER_ADDRESS'] = sellers_info_df['SELLER_ADDRESS'].astype(str)
    # Apply these transformations based on the 'PLATFORM' value
    is_aliexpress = sellers_info_df['PLATFORM'] == 'ALIEXPRESS'
    # Apply transformations for 'ALIEXPRESS' platform
    sellers_info_df.loc[is_aliexpress, 'SELLER_ADDRESS'] = sellers_info_df.loc[is_aliexpress, 'SELLER_ADDRESS']
    # Apply transformations for other platforms
    sellers_info_df.loc[~is_aliexpress, 'SELLER_ADDRESS'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_ADDRESS_CN'].apply(translate_to_english)
    sellers_info_df['SELLER_CITY'] = sellers_info_df['SELLER_CITY_CN'].apply(translate_to_english)
    sellers_info_df['SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE_CN'].apply(translate_to_english)
   
    # Sample DataFrame
    # Replace '-' in COMPANY_TYPE with 'Limited liability company' where COMPANY_NAME contains 'Co., Ltd'
    sellers_info_df.loc[sellers_info_df['COMPANY_TYPE'] == '-' & sellers_info_df['SELLER_BUSINESS_NAME'].str.contains('Co., Ltd'), 'COMPANY_TYPE'] = 'Limited liability company'
    sellers_info_df = sellers_info_df[['SHOP_NAMEextracted', 'SHOP_URLextracted', 'SELLER', 'SELLER_URL', "PLATFORM", "FILENAME", "SELLER_VAT_N", "SELLER_BUSINESS_NAME", "COMPANY_TYPE", "SELLER_ADDRESS", 'SELLER_PROVINCE', "SELLER_CITY", 'SELLER_EMAIL', 'SELLER_TEL_N', "LEGAL_REPRESENTATIVE", "ESTABLISHED_IN", "INITIAL_CAPITAL", "EXPIRATION_DATE", 'AIQICHA_URL', "SELLER_BUSINESS_NAME_CN", "COMPANY_TYPE_CN", "SELLER_ADDRESS_CN", 'SELLER_PROVINCE_CN', 'SELLER_CITY_CN',  "LEGAL_REPRESENTATIVE_CN", "BUSINESS_DESCRIPTION",  "REGISTRATION_INSTITUTION"]]

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






