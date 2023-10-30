import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import pandas as pd
import os
import datetime

# ...

# Define a function to generate a timestamp
def generate_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def remove_whitespace(df, columns):
    for column in columns:
        df[column] = df[column].str.replace(r'\s', '', regex=True)


from googletrans import Translator

# Create a translator instance
translator = Translator()

# Define a function to translate text
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='zh-CN', dest='en')
        return translated.text
    except Exception as e:
        return text


# Set the page title
st.set_page_config(page_title='Chinese Text OCR', layout='wide')

# Create a Streamlit app
st.title('Chinese Platform - Sellers Business Licence OCR Reader')
uploaded_images = st.file_uploader('Upload JPG or PNG images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Create a DataFrame to store the extracted text
df_sellers_info = pd.DataFrame()


if uploaded_images:
    # Initialize a dictionary to store the extracted text for each target
    # extracted_data = {}
    df_extraction = pd.DataFrame()

    for uploaded_image in uploaded_images:
    # # Display the uploaded image
    #     st.image(uploaded_image, use_column_width=True, caption='Uploaded Image')

        # Perform OCR on the entire uploaded image
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim')


        # Initialize a dictionary for this image
        extracted_data_per_image = {}        
        extracted_data_per_image['FILENAME'] = uploaded_image.name

        # Check if 'm.1688' is in the OCR text
        if 'scportaltaobao' in ocr_text:
            platform = 'TAOBAO'
        elif 'm.1688' in ocr_text:
            platform = 'CHINAALIBABA'
        elif '天猫网' or 'tmal' in ocr_text:
            platform = 'TMALL'
        else:  
            platform = None

        # Initialize a dictionary for this image
        extracted_data_per_image = {'PLATFORM': platform}
        # Add the filename to the dictionary
        extracted_data_per_image['FILENAME'] = uploaded_image.name
        # Create a selection box to allow the user to select the platform
        # Define the platform options
        platform_options = ['TMALL', 'TAOBAO', 'CHINAALIBABA']
        
        # Create a selection box with a unique key based on the filename
        selected_platform = st.selectbox(f'Select the platform for {uploaded_image.name}', platform_options, index=platform_options.index(extracted_data_per_image['PLATFORM']))
        # Update the PLATFORM value with the selected platform
        extracted_data_per_image['PLATFORM'] = selected_platform
        
        # Display a small preview of the uploaded image
        st.image(uploaded_image, use_column_width=False, caption=f'Uploaded Image: {uploaded_image.name}', width=400)


        if extracted_data_per_image['PLATFORM'] == 'TMALL' or None:
            # List of target texts
            targets = ['企业注册号', '企业名称', '类 型', '类 ”型', '类 。 型', '住所', '住 所', '住 ”所', '法定代表人', '成立时间', '注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
        if extracted_data_per_image['PLATFORM'] == 'TAOBAO':
            # Specify the page segmentation mode (PSM) as 6 for LTR and TTB reading
            ocr_text = pytesseract.image_to_string(image, lang='chi_sim', config='--psm 6')
            # List of target texts
            targets = ['注册号', '公司名称', '类型',  '地址', '法定代表人', '经营期限自', '注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
        if extracted_data_per_image['PLATFORM'] == 'CHINAALIBABA':
            # List of target texts
            targets = ['统一社会', '公司名称', '企业类型', '类 ”型', '类 。 型', '地址', '法定代表人', '成立日期', '注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
        
        # # Display the entire extracted text
        # st.subheader("Entire Extracted Text in Chinese")
        # st.write(ocr_text)

        for word in targets:
            # Find the coordinates where the text is found
            target_text = word
            text_location = [(m.start(0), m.end(0)) for m in re.finditer(target_text, ocr_text)]

            if text_location:
                if extracted_data_per_image['PLATFORM'] == 'TMALL' or None:
                    # Shift the coordinates to the right by 7 pixels
                    roi_data = [(start + 7, end + 50) for start, end in text_location]
                if extracted_data_per_image['PLATFORM'] == 'TAOBAO':
                    # Shift the coordinates to the right by 10 pixels
                    roi_data = [(start + 10, end + 150) for start, end in text_location]
                if extracted_data_per_image['PLATFORM'] == 'CHINAALIBABA':
                    # Shift the coordinates to the right by 10 pixels
                    roi_data = [(start + 10, end + 150) for start, end in text_location]# Extract text using roi_data
                extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                # st.write(f"Extracted Text for '{target_text}' (Shifted by 7 pixels to the right):")
                # st.write(extracted_texts)
                # Assign the extracted text to the dictionary
                extracted_data_per_image[word] = extracted_texts
            else:
                # If target text is not found, assign an empty list
                extracted_data_per_image[word] = ['']
        
        # Create a DataFrame for the extracted data of this image
        df_extraction_image = pd.DataFrame(extracted_data_per_image)
        # Append the DataFrame for this image to the main df_extraction
        df_extraction = pd.concat([df_extraction, df_extraction_image], ignore_index=True)


    # Copy the DataFrame
    df_sellers_info = df_extraction.copy()
    
    # st.subheader("Extracted Data")
    # st.dataframe(df_sellers_info)


    tmall_df = df_sellers_info[df_sellers_info['PLATFORM'].isin(['TMALL', None])]
    taobao_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'TAOBAO']
    chinaalibaba_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'CHINAALIBABA']


# ------------------------------------------------------------
#                             TMALL
# ------------------------------------------------------------


    # # Check for missing values and replace them with an empty string
    tmall_df['企业注册号'].fillna('', inplace=True)
    tmall_df['SELLER_VAT_N'] = tmall_df['企业注册号'].str.split('企业').str[0]
    tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.split('/').str[0]
    # # Remove all white spaces in the '企业注册号' column
    tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
    tmall_df.drop(['企业注册号'], axis=1, inplace=True)

    # # Split the 'SELLER_BUSINESS_NAME' column on ' 企业' and keep the part before it
    tmall_df['SELLER_BUSINESS_NAME'] = tmall_df['企业名称'].str.split('类').str[0]
    # # Remove all white spaces in the 'SELLER_VAT_N' column
    tmall_df['SELLER_BUSINESS_NAME'] = tmall_df['SELLER_BUSINESS_NAME'].str.replace(r'\s', '', regex=True)
    tmall_df.drop(['企业名称'], axis=1, inplace=True)
    
    tmall_df['COMPANY_TYPE'] = tmall_df['类 型'].fillna('') + tmall_df['类 ”型'].fillna('') + tmall_df['类 。 型'].fillna('')
    tmall_df['COMPANY_TYPE'] = tmall_df['COMPANY_TYPE'].str.split('住 ').str[0]
    tmall_df['COMPANY_TYPE'] = tmall_df['COMPANY_TYPE'].str.split('住 ').str[0]
    tmall_df['COMPANY_TYPE'] = tmall_df['COMPANY_TYPE'].str.split('|').str[0]
    tmall_df.drop(['类 型', '类 ”型', '类 。 型'], axis=1, inplace=True)

    tmall_df['SELLER_ADDRESS'] = tmall_df['住 所'].fillna('') + tmall_df['住 ”所'].fillna('') + tmall_df['住所'].fillna('')
    tmall_df['SELLER_ADDRESS'] = tmall_df['SELLER_ADDRESS'].str.split('法定').str[0]
    tmall_df['SELLER_ADDRESS'] = tmall_df['SELLER_ADDRESS'].str.split('|').str[0]
    tmall_df['SELLER_ADDRESS'] = tmall_df['SELLER_ADDRESS'].str.upper()
    tmall_df.drop(['住 所', '住 ”所', '住所'], axis=1, inplace=True)

    tmall_df['LEGAL_REPRESENTATIVE'] = tmall_df['法定代表人'].str.split('成').str[0]
    tmall_df['LEGAL_REPRESENTATIVE'] = tmall_df['LEGAL_REPRESENTATIVE'].str.split('|').str[0]
    tmall_df['LEGAL_REPRESENTATIVE'] = tmall_df['LEGAL_REPRESENTATIVE'].str.replace(r'\s', '', regex=True)

    tmall_df.drop(['法定代表人'], axis=1, inplace=True)

    tmall_df['BUSINESS_DESCRIPTION'] = tmall_df['经营范围'].fillna('') + tmall_df['经营学围'].fillna('')
    tmall_df.drop(['经营范围', '经营学围'], axis=1, inplace=True)

    tmall_df['ESTABLISHED_IN'] = tmall_df['成立时间'].str.split('注').str[0]
    tmall_df['ESTABLISHED_IN'] = tmall_df['ESTABLISHED_IN'].str.split('|').str[0]
    tmall_df.drop(['成立时间'], axis=1, inplace=True)

    tmall_df['INITIAL_CAPITAL'] = tmall_df['注册资本'].str.split('营').str[0]
    tmall_df['INITIAL_CAPITAL'] = tmall_df['INITIAL_CAPITAL'].str.split('|').str[0]
    tmall_df.drop(['注册资本'], axis=1, inplace=True)

    #'注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
    tmall_df['EXPIRATION_DATE'] = tmall_df['营业期限'].str.split('经').str[0]
    tmall_df['EXPIRATION_DATE'] = tmall_df['EXPIRATION_DATE'].str.split('|').str[0]
    tmall_df.drop(['营业期限'], axis=1, inplace=True)

    tmall_df['REGISTRATION_INSTITUTION'] = tmall_df['登记机关'].str.split('核').str[0]
    tmall_df['REGISTRATION_INSTITUTION'] = tmall_df['REGISTRATION_INSTITUTION'].str.split('|').str[0]
    tmall_df.drop(['登记机关'], axis=1, inplace=True)
    
    # st.dataframe(tmall_df)

# ------------------------------------------------------------
#                             TAOBAO
# ------------------------------------------------------------
# # Check for missing values and replace them with an empty string
    taobao_df['注册号'].fillna('', inplace=True)
    taobao_df['SELLER_VAT_N'] = taobao_df['注册号'].str.split('注册').str[0]
    taobao_df['SELLER_VAT_N'] = taobao_df['SELLER_VAT_N'].str.split('/').str[0]
    # # Remove all white spaces in the '企业注册号' column
    taobao_df['SELLER_VAT_N'] = taobao_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
    taobao_df.drop(['注册号'], axis=1, inplace=True)
    taobao_df.drop(['企业注册号'], axis=1, inplace=True)

    # # Split the 'SELLER_BUSINESS_NAME' column on ' 企业' and keep the part before it
    taobao_df['SELLER_BUSINESS_NAME'] = taobao_df['公司名称'].str.split('统一').str[0]
    # Remove all white spaces in the 'SELLER_VAT_N' column
    taobao_df['SELLER_BUSINESS_NAME'] = taobao_df['SELLER_BUSINESS_NAME'].str.replace(r'\s', '', regex=True)
    taobao_df.drop(['公司名称'], axis=1, inplace=True)
    taobao_df.drop(['企业名称'], axis=1, inplace=True)
    taobao_df['COMPANY_TYPE'] = taobao_df['类型'].str.split('经营').str[0]
    taobao_df['COMPANY_TYPE'] = taobao_df['COMPANY_TYPE'].str.replace(r'\s', '', regex=True)    
    taobao_df.drop(['类型', '类 ”型', '类 型', '类 。 型'], axis=1, inplace=True)

    taobao_df['SELLER_ADDRESS'] = taobao_df['地址'].str.split('法定').str[0]
    taobao_df['SELLER_ADDRESS'] = taobao_df['SELLER_ADDRESS'].str.upper()
    taobao_df['SELLER_ADDRESS'] = taobao_df['SELLER_ADDRESS'].str.replace(r'\s', '', regex=True)    
    taobao_df.drop(['地址', '住 所', '住所'], axis=1, inplace=True)

    taobao_df['LEGAL_REPRESENTATIVE'] = taobao_df['法定代表人'].str.split('公司').str[0]
    taobao_df['LEGAL_REPRESENTATIVE'] = taobao_df['LEGAL_REPRESENTATIVE'].str.replace(r'\s', '', regex=True)    
    taobao_df.drop(['法定代表人'], axis=1, inplace=True)

    taobao_df['BUSINESS_DESCRIPTION'] = taobao_df['经营范围'].fillna('') + taobao_df['经营学围'].fillna('')
    taobao_df.drop(['经营范围', '经营学围'], axis=1, inplace=True)

    taobao_df['ESTABLISHED_IN'] = taobao_df['经营期限自'].str.split('经').str[0]
    taobao_df['ESTABLISHED_IN'] = taobao_df['ESTABLISHED_IN'].str.replace(r'\s', '', regex=True)    
    taobao_df.drop(['成立时间'], axis=1, inplace=True)

    taobao_df['INITIAL_CAPITAL'] = taobao_df['注册资本'].str.split('营').str[0]
    taobao_df['INITIAL_CAPITAL'] = taobao_df['INITIAL_CAPITAL'].str.split('|').str[0]
    taobao_df.drop(['注册资本'], axis=1, inplace=True)

    #'注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
    taobao_df['EXPIRATION_DATE'] = taobao_df['营业期限'].str.split('经').str[0]
    taobao_df['EXPIRATION_DATE'] = taobao_df['EXPIRATION_DATE'].str.replace(r'\s', '', regex=True)    
    taobao_df.drop(['营业期限'], axis=1, inplace=True)

    taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['登记机关'].str.split('注').str[0]
    taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['REGISTRATION_INSTITUTION'].str.replace(r'\s', '', regex=True)        
    taobao_df.drop(['登记机关'], axis=1, inplace=True)
    
    # st.dataframe(taobao_df)
    
# ------------------------------------------------------------
#                             1688
# ------------------------------------------------------------

    # # Check for missing values and replace them with an empty string
    chinaalibaba_df['统一社会'].fillna('', inplace=True)
    chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['统一社会'].str.split('言').str[0]
    # # Remove all white spaces in the '企业注册号' column
    chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
    chinaalibaba_df.drop(['企业注册号'], axis=1, inplace=True)

    # # Split the 'SELLER_BUSINESS_NAME' column on ' 企业' and keep the part before it
    chinaalibaba_df['SELLER_BUSINESS_NAME'] = chinaalibaba_df['公司名称'].str.split('注').str[0]
    # # Remove all white spaces in the 'SELLER_VAT_N' column
    chinaalibaba_df['SELLER_BUSINESS_NAME'] = chinaalibaba_df['SELLER_BUSINESS_NAME'].str.replace(r'\s', '', regex=True)
    chinaalibaba_df.drop(['企业名称'], axis=1, inplace=True)
    
    chinaalibaba_df['COMPANY_TYPE'] = chinaalibaba_df['法定代表人'].str.split('经营').str[0]
    chinaalibaba_df['COMPANY_TYPE'] = chinaalibaba_df['COMPANY_TYPE'].str.replace(r'\s', '', regex=True)
    chinaalibaba_df.drop(['类 型', '类 ”型', '类 。 型'], axis=1, inplace=True)

    chinaalibaba_df['SELLER_ADDRESS'] = chinaalibaba_df['地址'].str.split('成立').str[0]
    chinaalibaba_df['SELLER_ADDRESS'] = chinaalibaba_df['SELLER_ADDRESS'].str.split('|').str[0]
    chinaalibaba_df['SELLER_ADDRESS'] = chinaalibaba_df['SELLER_ADDRESS'].str.replace(r'\s', '', regex=True)
    chinaalibaba_df['SELLER_ADDRESS'] = chinaalibaba_df['SELLER_ADDRESS'].str.upper()
    chinaalibaba_df.drop(['住 所', '住 ”所', '住所'], axis=1, inplace=True)

    chinaalibaba_df['LEGAL_REPRESENTATIVE'] = chinaalibaba_df['地址'].str.split('企业').str[0]        
    chinaalibaba_df['LEGAL_REPRESENTATIVE'] = chinaalibaba_df['LEGAL_REPRESENTATIVE'].str.split('表人').str[1]
    chinaalibaba_df['LEGAL_REPRESENTATIVE'] = chinaalibaba_df['LEGAL_REPRESENTATIVE'].str.replace(r'\s', '', regex=True)
    chinaalibaba_df.drop(['法定代表人'], axis=1, inplace=True)

    chinaalibaba_df['BUSINESS_DESCRIPTION'] = chinaalibaba_df['经营范围'].fillna('') + chinaalibaba_df['经营学围'].fillna('')
    chinaalibaba_df.drop(['经营范围', '经营学围'], axis=1, inplace=True)

    chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['成立时间'].str.split('注').str[0]
    chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['ESTABLISHED_IN'].str.split('|').str[0]
    chinaalibaba_df.drop(['成立时间'], axis=1, inplace=True)

    chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['注册资本'].str.split('营').str[0]
    chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['INITIAL_CAPITAL'].str.split('|').str[0]
    chinaalibaba_df.drop(['注册资本'], axis=1, inplace=True)

    #'注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
    chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['营业期限'].str.split('经').str[0]
    chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['EXPIRATION_DATE'].str.split('|').str[0]
    chinaalibaba_df.drop(['营业期限'], axis=1, inplace=True)

    chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['登记机关'].str.split('营业').str[0]
    chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['REGISTRATION_INSTITUTION'].str.split('|').str[0]
    chinaalibaba_df.drop(['登记机关'], axis=1, inplace=True)
    
    # st.dataframe(chinaalibaba_df)    
    

   
    
    # Concatenate them into a single DataFrame
    sellers_info_df = pd.concat([tmall_df, taobao_df, chinaalibaba_df], ignore_index=True)
    sellers_info_df.drop(['统一社会', '公司名称', '企业类型', '地址', '成立日期', '注册号', '类型', '住 ”所', '经营期限自'], axis=1, inplace=True)
    sellers_info_df['AIQICHA_URL'] = 'https://www.aiqicha.com/s?q=' + sellers_info_df['SELLER_VAT_N']
    sellers_info_df.drop(['该准时间'], axis=1, inplace=True)
    # Apply the translation function to the 'SELLER_BUSINESS_NAME' column
    sellers_info_df['SELLER_BUSINESS_NAME_EN'] = sellers_info_df['SELLER_BUSINESS_NAME'].apply(translate_to_english)
    sellers_info_df['COMPANY_TYPE_EN'] = sellers_info_df['COMPANY_TYPE'].apply(translate_to_english)
    sellers_info_df['LEGAL_REPRESENTATIVE_EN'] = sellers_info_df['LEGAL_REPRESENTATIVE'].apply(translate_to_english)
    sellers_info_df['SELLER_ADDRESS_EN'] = sellers_info_df['SELLER_ADDRESS'].apply(translate_to_english)
    columns_to_remove_whitespace = ['COMPANY_TYPE']
    remove_whitespace(sellers_info_df, columns_to_remove_whitespace)
    
    sellers_info_df = sellers_info_df[["PLATFORM", "FILENAME", "SELLER_VAT_N", "SELLER_BUSINESS_NAME", "COMPANY_TYPE", "SELLER_ADDRESS", "LEGAL_REPRESENTATIVE", "AIQICHA_URL", "SELLER_BUSINESS_NAME_EN", "COMPANY_TYPE_EN", "LEGAL_REPRESENTATIVE_EN", "SELLER_ADDRESS_EN", "BUSINESS_DESCRIPTION", "ESTABLISHED_IN", "INITIAL_CAPITAL", "EXPIRATION_DATE", "REGISTRATION_INSTITUTION"]]

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






