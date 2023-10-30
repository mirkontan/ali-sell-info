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
st.markdown("<h1 style='text-align: center;'>Chinese Platform - Sellers Business Licence OCR Reader</h1>", unsafe_allow_html=True)

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
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim')
       

        if not uploaded_xlsx:
            # !!!! TENERE TMALL SEMPRE PER ULTIMA !!!!! 
            # ALTRIMENTI NON RICONOSCE LE ALTRE PIATTAFORME (non capisco perché)    
            # Check platform in the OCR text
            st.write(ocr_text)
            if 'scportaltaobao' in ocr_text:
                platform = 'TAOBAO'
            if '京东商城' or 'malljd' in ocr_text:
                platform = 'JD COM'
            if 'm.1688' in ocr_text:
                platform = 'CHINAALIBABA'
            if '天猫网' in ocr_text:
                platform = 'TMALL'
            if 'Informazioni' in ocr_text:
                platform = 'ALIEXPRESS'
           
        else:
            row = df_urls.iloc[i]  # We want to match with the first row of df_urls
            extracted_data_per_image['SELLER'] = row['SELLER']
            extracted_data_per_image['SELLER_URL'] = row['SELLER_URL']
            platform = row['PLATFORM']


        # Assign the platform to the dictionary for this image
        extracted_data_per_image['PLATFORM'] = platform
        
        # Create a selection box to allow the user to select the platform
        # Define the platform options
        platform_options = ['TMALL', 'TAOBAO', 'CHINAALIBABA', 'JD COM', 'ALIEXPRESS', 'TEST']
        
        # Create a selection box with a unique key based on the filename
        selected_platform = st.selectbox(f'Select the platform for {uploaded_image.name}', platform_options, index=platform_options.index(extracted_data_per_image['PLATFORM']), key=f"selectbox_{i}")
        # Update the PLATFORM value with the selected platform
        extracted_data_per_image['PLATFORM'] = selected_platform
        
        # Display a small preview of the uploaded image
        st.image(uploaded_image, use_column_width=False, caption=f'Uploaded Image: {uploaded_image.name}', width=400)
        
        # Set the target words to look up in each image
        targets_tmall = ['企业注册号', '企业名称', '类 型', '类 ”型', '类 。 型', '住所', '住 所', '住 ”所', '法定代表人', '成立时间', '注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
        targets_taobao = ['注册号', '公司名称', '类型',  '地址', '法定代表人', '经营期限自', '注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
        targets_chinaalibaba = ['统一社会', '公司名称', '企业类型', '类 ”型', '类 。 型', '地址', '法定代表人', '成立日期', '注册资本', '营业期限', '经营范围', '经营学围', '登记机关', '该准时间']
        targets_jd = ['卖家', '卖 家', '企业名称', '注册号', '注则号', '所在地', '地址', '网址', '法定代表人', '注册资本', '有效期', '经营范围', '经营学围', '店铺名称']
        targets_aliexpress = ['卖家', '卖 家', '企业名称', '注册号', '注则号', '所在地', '地址', '网址', '法定代表人', '注册资本', '有效期', '经营范围', '经营学围', '店铺名称']
        targets_test = ['企业注册号', '注册号', '企业名称', '公司名称', '类 型', '类 ”型', '类 。 型', '类型', '地址', '住所', '住 所', '住 ”所']

        if extracted_data_per_image['PLATFORM'] == 'TMALL':
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='chi_sim', config='--psm 6')
            st.write('color image ocr text')
            st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            st.write(lines)
            data = {}
            current_key = None

            for line in lines:
                # Split based on a colon (":") or semicolon (";")
                parts = re.split(r'[:;]', line, 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    data[key] = value
                    current_key = key
                else:
                    # If no colon or semicolon found, append to the previous key if available
                    if current_key is not None:
                        data[current_key] += ' ' + line.strip()          
            # Assign the 'PLATFORM' and 'FILENAME' values in the extracted_data_per_image dictionary
            updated_data = {}
            for key, value in data.items():
                if '企业注册号' in key:
                    key = '企业注册号'
                updated_data[key] = value

            # Replace the original data dictionary with the updated one
            data = updated_data

            extracted_data_per_image_tmall = pd.DataFrame([data])
            extracted_data_per_image_tmall['PLATFORM'] = 'TMALL'
            extracted_data_per_image_tmall['FILENAME'] = uploaded_image.name
            extracted_data_per_image_tmall['FULL_TEXT'] = ocr_text
            st.write(extracted_data_per_image_tmall)
            st.write(data)
            df_extraction_tmall = pd.concat([df_extraction_tmall, extracted_data_per_image_tmall], ignore_index=True)

            targets = targets_tmall
            
        if extracted_data_per_image['PLATFORM'] == 'TAOBAO':
            # Specify the page segmentation mode (PSM) as 6 for LTR and TTB reading
            ocr_text = pytesseract.image_to_string(image, lang='chi_sim', config='--psm 6')
            # List of target texts
            targets = targets_taobao
        if extracted_data_per_image['PLATFORM'] == 'CHINAALIBABA':
            # List of target texts
            targets = targets_chinaalibaba
        if extracted_data_per_image['PLATFORM'] == 'JD COM':
            # List of target texts
            targets = targets_jd
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
        






            # # Crop the grayscale image using OpenCV
            # # Define the coordinates for cropping (left, top, width, height)
            # left = 125
            # top = 80
            # width = 500  # Adjust based on your needs
            # height = 300  # Adjust based on your needs
            # cropped_image = grayscale_image[top:top+height, left:left+width]
            # # Now you can perform OCR on the grayscale image using pytesseract
            # ocr_text = pytesseract.image_to_string(cropped_image, lang='eng')
            # st.image(cropped_image)
            # st.write('grey image ocr text')
            # st.write(ocr_text)
     

            # # Split the text into lines
            # lines = ocr_text.splitlines()
            # # Split the text into lines
            # lines = ocr_text.splitlines()
            # st.write(lines)
            # data = {}
            # for line in lines:
            #     parts = line.split()
            #     if len(parts) == 2:
            #         key, value = parts
            #         data[key] = value
            # st.write(data)


            # List of target texts

        if extracted_data_per_image['PLATFORM'] == 'TEST' or None:
            targets = targets_test

        # Display the entire extracted text
        st.subheader("Entire Extracted Text in Chinese")
        st.write(ocr_text)

        for word in targets:
            # Find the coordinates where the text is found
            target_text = word
            text_location = [(m.start(0), m.end(0)) for m in re.finditer(target_text, ocr_text)]

            if text_location:
                extracted_texts = ''
                if extracted_data_per_image['PLATFORM'] == 'TMALL' or None:
                    # # Shift the coordinates to the right by 7 pixels
                    # roi_data = [(start + 4, end + 50) for start, end in text_location]
                    # extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                    pass
                elif extracted_data_per_image['PLATFORM'] == 'TAOBAO':
                    # Shift the coordinates to the right by 10 pixels
                    roi_data = [(start + 10, end + 150) for start, end in text_location]
                    extracted_texts = [ocr_text[start:end] for start, end in roi_data]
               
                elif extracted_data_per_image['PLATFORM'] == 'JD COM':
                    # if word == '京东商城网店':
                    #     # Shift the coordinates to the left by 50 pixels
                    #     roi_data = [(start - 50, end - 5) for start, end in text_location]
                    #     extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                    # else:
                    #     # Shift the coordinates to the right by 10 pixels
                    #     roi_data = [(start + 6, end + 150) for start, end in text_location]
                    #     extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                        # Shift the coordinates to the right by 10 pixels
                        roi_data = [(start + 6, end + 150) for start, end in text_location]
                        extracted_texts = [ocr_text[start:end] for start, end in roi_data]


                elif extracted_data_per_image['PLATFORM'] == 'CHINAALIBABA':
                    # Shift the coordinates to the right by 5 pixels
                    roi_data = [(start + 5, end + 150) for start, end in text_location]  # Extract text using roi_data
                    extracted_texts = [ocr_text[start:end] for start, end in roi_data]

                elif extracted_data_per_image['PLATFORM'] == 'TEST':
                    # Shift the coordinates to the right by 7 pixels
                    roi_data = [(start + 0, end + 50) for start, end in text_location]
                    extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                
                elif extracted_data_per_image['PLATFORM'] == 'ALIEXPRESS':
                    pass

                if extracted_texts:
                    st.write(f"Extracted Text for '{target_text}' (Shifted by pixels to the right):")
                    st.write(extracted_texts)
                    # Assign the extracted text to the dictionary
                    extracted_data_per_image[word] = extracted_texts
            else:
                # If target text is not found, assign an empty list
                extracted_data_per_image[word] = ['']
        
        # Create a DataFrame for the extracted data of this image
        df_extraction_image = pd.DataFrame(extracted_data_per_image)
        
        st.write( 'df_extraction_image')
        st.write(df_extraction_image)
        # Append the DataFrame for this image to the main df_extraction
        df_extraction = pd.concat([df_extraction, df_extraction_image], ignore_index=True)
    

    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'ALIEXPRESS']   
    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'TMALL']   
    df_extraction_overall = pd.concat([df_extraction, df_extraction_aliexpress], ignore_index=True)
    df_extraction_overall = pd.concat([df_extraction_overall, df_extraction_tmall], ignore_index=True)
    st.header('df extraXTION')
    st.write(df_extraction)
    st.header('df extraXTION ALIEXPRESS')
    st.write(df_extraction_aliexpress)
    st.header('df extraXTION TMALL')
    st.write(df_extraction_tmall)

    st.header('df extraXTION OVERALL')
    st.write(df_extraction_overall)

    # Copy the DataFrame
    df_sellers_info = df_extraction_overall.copy()

    tmall_df = df_sellers_info[df_sellers_info['PLATFORM'].isin(['TMALL', None])]
    taobao_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'TAOBAO']
    chinaalibaba_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'CHINAALIBABA']
    jd_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'JD COM']
    aliexpress_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'ALIEXPRESS']
    test_df = df_sellers_info[df_sellers_info['PLATFORM'] == 'TEST']
    


# ------------------------------------------------------------
#                             TMALL
# ------------------------------------------------------------

    if tmall_df.empty:
        # Create new columns based on the items in targets_taobao
        for target in targets_tmall:
            tmall_df[target] = None  # Add new columns with None values
    else:

        # # Check for missing values and replace them with an empty string
        tmall_df['企业注册号'].fillna('', inplace=True)
        tmall_df['SELLER_VAT_N'] = tmall_df['企业注册号']
        # tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.split('/').str[0]
        # tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.split(':').str[1]
            # Remove all white spaces in the '企业注册号' column
        tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].astype(str)
        tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)

        tmall_df['SELLER_BUSINESS_NAME_CN'] = tmall_df['企业名称']
        # tmall_df['SELLER_BUSINESS_NAME_CN'] = tmall_df['SELLER_BUSINESS_NAME_CN'].str.replace(r':', '', regex=False)
        # tmall_df['SELLER_BUSINESS_NAME_CN'] = tmall_df['SELLER_BUSINESS_NAME_CN'].str.replace(r';', '', regex=False)
        # tmall_df['SELLER_BUSINESS_NAME_CN'] = tmall_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'\s', '', regex=True)
        
        tmall_df['COMPANY_TYPE_CN'] = tmall_df['类 型'].fillna('') + tmall_df['类 ”型'].fillna('') + tmall_df['类 。 型'].fillna('')
        tmall_df['COMPANY_TYPE_CN'] = tmall_df['COMPANY_TYPE_CN'].str.split('住').str[0]
        tmall_df['COMPANY_TYPE_CN'] = tmall_df['COMPANY_TYPE_CN'].str.split('|').str[0]
        tmall_df['COMPANY_TYPE_CN'] = tmall_df['COMPANY_TYPE_CN'].str.replace(r'型', '', regex=False)
        tmall_df['COMPANY_TYPE_CN'] = tmall_df['COMPANY_TYPE_CN'].str.replace(r':', '', regex=False)
        tmall_df['COMPANY_TYPE_CN'] = tmall_df['COMPANY_TYPE_CN'].str.replace(r';', '', regex=False)


        tmall_df['SELLER_ADDRESS_CN'] = tmall_df['住所']
        # tmall_df['SELLER_ADDRESS_CN'] = tmall_df['SELLER_ADDRESS_CN'].str.split('法定').str[0]
        # tmall_df['SELLER_ADDRESS_CN'] = tmall_df['SELLER_ADDRESS_CN'].str.split('|').str[0]
        # tmall_df['SELLER_ADDRESS_CN'] = tmall_df['SELLER_ADDRESS_CN'].str.upper()
        # tmall_df['SELLER_ADDRESS_CN'] = tmall_df['SELLER_ADDRESS_CN'].str.replace(r':', '', regex=False)
        # tmall_df['SELLER_ADDRESS_CN'] = tmall_df['SELLER_ADDRESS_CN'].str.replace(r';', '', regex=False)
        # tmall_df['SELLER_ADDRESS_CN'] = tmall_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)

        tmall_df['LEGAL_REPRESENTATIVE_CN'] = tmall_df['法定代表人'].str.split('成').str[0]
        tmall_df['LEGAL_REPRESENTATIVE_CN'] = tmall_df['LEGAL_REPRESENTATIVE_CN'].str.split('|').str[0]
        tmall_df['LEGAL_REPRESENTATIVE_CN'] = tmall_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)
        tmall_df['LEGAL_REPRESENTATIVE_CN'] = tmall_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'人:', '', regex=False)
        tmall_df['LEGAL_REPRESENTATIVE_CN'] = tmall_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'人;', '', regex=False)



        tmall_df['BUSINESS_DESCRIPTION'] = tmall_df['经营范围'].fillna('') + tmall_df['经营学围'].fillna('')

        tmall_df['ESTABLISHED_IN'] = tmall_df['成立时间'].str.split('注').str[0]
        tmall_df['ESTABLISHED_IN'] = tmall_df['ESTABLISHED_IN'].str.split('|').str[0]

        tmall_df['INITIAL_CAPITAL'] = tmall_df['注册资本'].str.split('营').str[0]
        tmall_df['INITIAL_CAPITAL'] = tmall_df['INITIAL_CAPITAL'].str.split('|').str[0]

        tmall_df['EXPIRATION_DATE'] = tmall_df['营业期限'].str.split('至').str[1]

        tmall_df['REGISTRATION_INSTITUTION'] = tmall_df['登记机关'].str.split('核').str[0]
        tmall_df['REGISTRATION_INSTITUTION'] = tmall_df['REGISTRATION_INSTITUTION'].str.split('|').str[0]
        tmall_df['REGISTRATION_INSTITUTION'] = tmall_df['REGISTRATION_INSTITUTION'].str.split('注').str[0]
        tmall_df['SELLER_ADDRESS_CITY'] = '-'
        tmall_df['SHOP_NAMEextracted'] = '-'
        tmall_df['SHOP_URLextracted'] = '-'       
        st.header(tmall_df)
        st.dataframe(tmall_df)

# ------------------------------------------------------------
#                             TAOBAO
# ------------------------------------------------------------
   
    # Check if taobao_df is empty
    if taobao_df.empty:
        # Create new columns based on the items in targets_taobao
        for target in targets_taobao:
            taobao_df[target] = None  # Add new columns with None values
    else:
    # # Check for missing values and replace them with an empty string
        taobao_df['注册号'].fillna('', inplace=True)
        taobao_df['SELLER_VAT_N'] = taobao_df['注册号'].str.split('注册').str[0]
        taobao_df['SELLER_VAT_N'] = taobao_df['SELLER_VAT_N'].str.split('/').str[0]
            # Remove all white spaces in the '企业注册号' column
        taobao_df['SELLER_VAT_N'] = taobao_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)

        taobao_df['SELLER_BUSINESS_NAME_CN'] = taobao_df['公司名称'].str.split('统一').str[0]
        taobao_df['SELLER_BUSINESS_NAME_CN'] = taobao_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'\s', '', regex=True)

        taobao_df['COMPANY_TYPE'] = taobao_df['类型'].str.split('经营').str[0]
        taobao_df['COMPANY_TYPE'] = taobao_df['COMPANY_TYPE'].str.replace(r'\s', '', regex=True)    

        taobao_df['SELLER_ADDRESS_CN'] = taobao_df['地址'].str.split('法定').str[0]
        taobao_df['SELLER_ADDRESS_CN'] = taobao_df['SELLER_ADDRESS_CN'].str.upper()
        taobao_df['SELLER_ADDRESS_CN'] = taobao_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)    

        taobao_df['LEGAL_REPRESENTATIVE_CN'] = taobao_df['法定代表人'].str.split('公司').str[0]
        taobao_df['LEGAL_REPRESENTATIVE_CN'] = taobao_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)    

        taobao_df['BUSINESS_DESCRIPTION'] = taobao_df['经营范围'].fillna('') + taobao_df['经营学围'].fillna('')

        taobao_df['ESTABLISHED_IN'] = taobao_df['经营期限自'].str.split('经').str[0]
        taobao_df['ESTABLISHED_IN'] = taobao_df['ESTABLISHED_IN'].str.replace(r'\s', '', regex=True)    

        taobao_df['INITIAL_CAPITAL'] = taobao_df['注册资本'].str.split('营').str[0]
        taobao_df['INITIAL_CAPITAL'] = taobao_df['INITIAL_CAPITAL'].str.split('|').str[0]

        taobao_df['EXPIRATION_DATE'] = taobao_df['营业期限'].str.split('经').str[0]
        taobao_df['EXPIRATION_DATE'] = taobao_df['EXPIRATION_DATE'].str.replace(r'\s', '', regex=True)    

        taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['登记机关'].str.split('注').str[0]
        taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['REGISTRATION_INSTITUTION'].str.replace(r'\s', '', regex=True)        
        taobao_df['SELLER_ADDRESS_CITY'] = '-'
        taobao_df['SHOP_NAMEextracted'] = '-'
        taobao_df['SHOP_URLextracted'] = '-'
        # st.dataframe(taobao_df)
    
# ------------------------------------------------------------
#                             1688
# ------------------------------------------------------------
    if chinaalibaba_df.empty:
            # Create new columns based on the items in targets_taobao
            for target in targets_chinaalibaba:
                chinaalibaba_df[target] = None  # Add new columns with None values
    else:
        # # Check for missing values and replace them with an empty string
        chinaalibaba_df['统一社会'].fillna('', inplace=True)
        chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['统一社会'].str.split('注册号').str[1]
        chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.split('企业').str[0]
        chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.split('登记').str[0]
        chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].astype(str)
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.split('企业').str[0]
       
        # chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['公司名称'].str.split('注').str[0]
        chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['公司名称'].str.split('注册').str[0]
        # chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'\s', '', regex=True)
        
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['法定代表人'].str.split('类').str[1]
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.split('登记').str[0]
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.split('上册').str[0]
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'主', '', regex=True)
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'串', '', regex=True)
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'估', '', regex=True)
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'型', '', regex=True)
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'\s', '', regex=True)

        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['地址'].str.split('成立').str[0]
        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.split('|').str[0]
        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.split('注册').str[0]
        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)
        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.upper()

        chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['地址'].str.split('企业').str[0]        
        chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'].str.split('表人').str[1]
        chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'].str.split('注册').str[0]
        chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)

        chinaalibaba_df['BUSINESS_DESCRIPTION'] = chinaalibaba_df['经营范围'].fillna('') + chinaalibaba_df['经营学围'].fillna('')

        chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['注册资本'].str.split('成立日期').str[1]
        chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['ESTABLISHED_IN'].str.split('|').str[0]
        chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['ESTABLISHED_IN'].str.split('统一社会').str[0]
        chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['ESTABLISHED_IN'].str.strip()

        chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['注册资本'].str.split('营').str[0]
        chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['INITIAL_CAPITAL'].str.split('|').str[0]

        chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['营业期限'].str.split('经').str[0]
        chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['EXPIRATION_DATE'].str.split('|').str[0]
        chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['EXPIRATION_DATE'].str.replace(r'^.*至今', 'Active', regex=True)

        chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['登记机关'].str.split('营业').str[0]
        chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['REGISTRATION_INSTITUTION'].str.split('|').str[0]
        chinaalibaba_df['SELLER_ADDRESS_CITY'] = '-'
        chinaalibaba_df['SHOP_NAMEextracted'] = '-'
        chinaalibaba_df['SHOP_URLextracted'] = '-'
        # st.dataframe(chinaalibaba_df)    
  
# ------------------------------------------------------------
#                             JD COM
# ------------------------------------------------------------

    if jd_df.empty:
        # Create new columns based on the items in targets_taobao
        for target in targets_jd:
            jd_df[target] = None  # Add new columns with None values
    else:
 #     ['卖家', '卖 家', ' '', '店铺名称']

        # # Check for missing values and replace them with an empty string
        jd_df['SELLER_VAT_N'] = jd_df['注册号'].fillna('') + jd_df['注则号'].fillna('') 
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.split('丢定代').str[0]
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'法证代', '法定代', regex=True)
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.split('法定代').str[0]
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'\。', '', regex=True)
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.split(r'|').str[0]

            # Remove all white spaces in the '企业注册号' column
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].astype(str)
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)

        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['企业名称'].str.split('营业').str[0]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'\s', '', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.lstrip()
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split('有限公司').str[0] + '有限公司'
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split('综合').str[0]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.lstrip()

        jd_df['SELLER_ADDRESS_CITY'] = jd_df['所在地'].str.split('市').str[0] + '市'
        jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.lstrip()
        jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.replace(r'\。', '', regex=True)
       
        jd_df['SELLER_ADDRESS_CN'] = jd_df['地址'].str.split('店铺名称').str[0]
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.lstrip()
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.upper()
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)

        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['法定代表人'].str.split('名:').str[1]
        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].astype(str).str.lstrip()
        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.slice(0, 3)

        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)
        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'人:', '', regex=False)
        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'人;', '', regex=False)

        jd_df['SHOP_NAMEextracted'] = jd_df['店铺名称']
        jd_df['SHOP_NAMEextracted'] = jd_df['SHOP_NAMEextracted'].str.replace('店生网址', '店铺网址', regex=False)
        jd_df['SHOP_NAMEextracted'] = jd_df['SHOP_NAMEextracted'].str.split('店铺网').str[0]
        jd_df['SHOP_NAMEextracted'] = jd_df['SHOP_NAMEextracted'].str.lstrip()

        jd_df['SHOP_URLextracted'] = jd_df['网址']
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.split('html').str[0] + 'html'
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.replace(r'indexr', 'index-', regex=False)
        jd_df['SHOP_URLextracted'] = 'https://mall.jd.com/index-' + jd_df['SHOP_URLextracted'].str.split('index-').str[1]

        jd_df['BUSINESS_DESCRIPTION'] = jd_df['经营范围'].fillna('') + jd_df['经营学围'].fillna('')

        jd_df['ESTABLISHED_IN'] = jd_df['有效期'].str.split('注').str[0]
        jd_df['ESTABLISHED_IN'] = jd_df['ESTABLISHED_IN'].str.split('|').str[0]

        jd_df['INITIAL_CAPITAL'] = jd_df['注册资本'].str.split('营').str[0]
        jd_df['INITIAL_CAPITAL'] = jd_df['INITIAL_CAPITAL'].str.split('|').str[0]

        jd_df['EXPIRATION_DATE'] = jd_df['有效期'].str.split('经').str[0]
        jd_df['EXPIRATION_DATE'] = jd_df['EXPIRATION_DATE'].str.split('|').str[0]
        jd_df['COMPANY_TYPE_CN'] = '-'
        jd_df['REGISTRATION_INSTITUTION'] = '-'


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






