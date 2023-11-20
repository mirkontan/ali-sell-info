import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import pandas as pd
import os
import datetime
from PIL import Image
from io import BytesIO
import time
import platform as sys_platform  # Rename platform to avoid conflicts
from geo_dict import city_to_province
from geo_dict import country_city_dict
from geo_dict import country_area_dict

# from hk_company_crawl import crawl_hk_company_data  # Import the crawling function


# Define a function to generate a timestamp
def generate_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

def remove_whitespace(df, columns):
    for column in columns:
        df[column] = df[column].str.replace(r'\s', '', regex=True)



# Set the page title
st.set_page_config(page_title='Business Licence OCR Reader', layout='wide')

# Create a Streamlit app
st.markdown("<h1 style='text-align: center;'>Business Licence OCR Reader</h1>", unsafe_allow_html=True)

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



# Create a selection box to allow the user to select the platform
# Define the platform options
platform_options = ['TMALL', 'TAOBAO', 'CHINAALIBABA', 'JD COM', 'ALIEXPRESS', 'TEST']

# Create a selection box with a unique key based on the filename
selected_platform_all = st.selectbox(f'Select the platform FOR ALL IMAGES', platform_options, key=f"selectboxall_")


if uploaded_images:
    # Define a custom sorting key function
    def custom_sort_key(x):
        # Remove common image file extensions and then convert to lowercase for sorting
        filename, ext = os.path.splitext(x.name)
        filename = filename.lower()

        try:
            # Try to convert the filename to an integer (for purely numeric filenames)
            return (int(filename), ext, str(x.name))
        except ValueError:
            # If conversion to int fails, treat it as a string
            return (float('inf'), ext, str(x.name))
        
    # Sort the uploaded images by filename
    sorted_images = sorted(uploaded_images, key=custom_sort_key)

    # Initialize a dictionary to store the extracted text for each target
    # extracted_data = {}
    df_extraction = pd.DataFrame()
    df_extraction_aliexpress = pd.DataFrame()
    df_extraction_tmall = pd.DataFrame()
    df_extraction_jd = pd.DataFrame()
    df_extraction_taobao = pd.DataFrame()
    df_extraction_chinaalibaba = pd.DataFrame()


    for i, uploaded_image in enumerate(sorted_images):
        # Display the uploaded image
        # st.image(uploaded_image, use_column_width=True, caption='Uploaded Image')

        # Initialize a dictionary for this image
        extracted_data_per_image = {'SELLER': None, 'SELLER_URL': None}
        # Add the filename to the dictionary
        extracted_data_per_image['FILENAME'] = uploaded_image.name

        # Perform OCR on the entire uploaded image (CN language)
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        ocr_text = pytesseract.image_to_string(image, lang='eng')
       
        if selected_platform_all:
            platform = selected_platform_all
        else: 
            if not uploaded_xlsx:
                # !!!! TENERE TMALL SEMPRE PER ULTIMA !!!!! 
                # ALTRIMENTI NON RICONOSCE LE ALTRE PIATTAFORME (non capisco perché)    
                # Check platform in the OCR text
                # st.write(ocr_text)
                if '网店经营者' in ocr_text:
                    platform = 'TAOBAO'
                if '京东商城' or 'malljd' in ocr_text:
                    platform = 'JD COM'
                if '主体资质' in ocr_text:
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
        # platform_options = ['TMALL', 'TAOBAO', 'CHINAALIBABA', 'JD COM', 'ALIEXPRESS', 'TEST']
        
        # Create a selection box with a unique key based on the filename
        selected_platform = st.selectbox(f'Select the platform for {uploaded_image.name}', platform_options, index=platform_options.index(extracted_data_per_image['PLATFORM']), key=f"selectbox_{i}")
        # Update the PLATFORM value with the selected platform
        extracted_data_per_image['PLATFORM'] = selected_platform
        
        # Display a small preview of the uploaded image
        st.image(uploaded_image, use_column_width=False, caption=f'Uploaded Image: {uploaded_image.name}', width=400)

        
        if extracted_data_per_image['PLATFORM'] == 'TMALL':
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            # st.write('color image ocr text')
            # st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            # st.write(lines)
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
            
        if extracted_data_per_image['PLATFORM'] == 'TAOBAO':
            # Specify the page segmentation mode (PSM) as 6 for LTR and TTB reading
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            # st.write('color image ocr text')
            # st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            # st.write(lines)
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

            extracted_data_per_image_taobao = pd.DataFrame([data])
            extracted_data_per_image_taobao['PLATFORM'] = 'TAOBAO'
            extracted_data_per_image_taobao['FILENAME'] = uploaded_image.name
            extracted_data_per_image_taobao['FULL_TEXT'] = ocr_text
            st.write(extracted_data_per_image_taobao)
            st.write(data)
            df_extraction_taobao = pd.concat([df_extraction_taobao, extracted_data_per_image_taobao], ignore_index=True)

        if extracted_data_per_image['PLATFORM'] == 'CHINAALIBABA':
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Perform OCR on the entire uploaded image (IT language)
            ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

            # Determine the split point based on the image width (assuming an even split)
            split_point = image.shape[1] // 2  # Split horizontally in the middle of the image

            # Split the image into left and right sections
            left_image = image[:, :split_point]
            right_image = image[:, split_point:]

            # Perform OCR on the left and right sections
            left_ocr_text = pytesseract.image_to_string(left_image, lang='eng', config='--psm 6')
            right_ocr_text = pytesseract.image_to_string(right_image, lang='eng', config='--psm 6')
            # st.write(left_ocr_text)
            # st.write(right_ocr_text)
            # Process left and right columns separately
            # Here, you can parse and extract data from each column as needed

            # Example for the left column
            # Exam# Example for the left column
            left_data_list = []
            right_data_list = []
            left_column_keys = []
            right_column_keys = []

            # Example for the left column
            left_lines = left_ocr_text.splitlines()

            for line in left_lines:
                # Split based on a whitespace (" ")
                parts = re.split(' ', line, 1)
                if len(parts) >= 2:
                    key = parts[0]
                    value = ' '.join(parts[1:])
                    left_column_keys.append(key)
                    left_data_list.append(value)

            # Example for the right column
            right_lines = right_ocr_text.splitlines()

            for line in right_lines:
                # Split based on a whitespace (" ")
                parts = re.split(' ', line, 1)
                if len(parts) >= 2:
                    key = parts[0]
                    value = ' '.join(parts[1:])
                    right_column_keys.append(key)
                    right_data_list.append(value)

            # Determine the maximum number of columns needed for both left and right columns
            max_left_columns = len(left_column_keys)
            max_right_columns = len(right_column_keys)

            # Pad the data lists with empty strings to match the number of columns
            while len(left_data_list) < max_left_columns:
                left_data_list.append('')
            while len(right_data_list) < max_right_columns:
                right_data_list.append('')

            # Create DataFrames for the left and right columns
            extracted_data_left = pd.DataFrame([left_data_list])
            extracted_data_right = pd.DataFrame([right_data_list])

            # Rename the columns in the left DataFrame with left column keys
            extracted_data_left.columns = left_column_keys

            # Rename the columns in the right DataFrame with right column keys
            extracted_data_right.columns = right_column_keys
            
            # st.write(extracted_data_left)
            # st.write(extracted_data_right)

            # Concatenate data row-wise (vertically)
            extracted_data_per_image_chinaalibaba = pd.concat([extracted_data_left, extracted_data_right], axis=1)
            # st.write(extracted_data_per_image_chinaalibaba)

            # Add additional columns as needed
            extracted_data_per_image_chinaalibaba['PLATFORM'] = 'CHINAALIBABA'
            extracted_data_per_image_chinaalibaba['FILENAME'] = uploaded_image.name
            extracted_data_per_image_chinaalibaba['FULL_TEXT'] = ocr_text

            # Concatenate the result with an existing DataFrame if required
            df_extraction_chinaalibaba = pd.concat([df_extraction_chinaalibaba, extracted_data_per_image_chinaalibaba], ignore_index=True)

        if extracted_data_per_image['PLATFORM'] == 'JD COM':
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            # st.write('color image ocr text')
            # st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            # st.write(lines)

            data = {}
            current_key = None

            # Define a regular expression pattern for matching Chinese characters
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')

            for line in lines:
                # Split based on a colon (":") or semicolon (";")
                parts = re.split(r'[:;]', line, 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                  
                    # Check if the value contains Chinese characters
                    if key == '企业名称':
                        if re.search(chinese_pattern, value):
                            data[key] = value  # Use Chinese language settings for this key's value
                        else:
                            # Use English language settings for this key's value
                            data[key] = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                            data[key] = data[key].split(':', 1)[1].strip()

                    else:
                        data[key] = value
                        
                    current_key = key
                else:
                    # If no colon or semicolon found, append to the previous key if available
                    if current_key is not None:
                        data[current_key] += ' ' + line.strip()          

            
            extracted_data_per_image_jd = pd.DataFrame([data])
            extracted_data_per_image_jd['PLATFORM'] = 'JD COM'
            extracted_data_per_image_jd['FILENAME'] = uploaded_image.name
            extracted_data_per_image_jd['FULL_TEXT'] = ocr_text
            # st.write(extracted_data_per_image_jd)
            # st.header('JD DATA EXTRACTED IN DICTIONARY')
            # st.write(data)
            df_extraction_jd = pd.concat([df_extraction_jd, extracted_data_per_image_jd], ignore_index=True)



        if extracted_data_per_image['PLATFORM'] == 'ALIEXPRESS':
            # Perform OCR on the entire uploaded image (IT language)
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            #test - COLOR IMAGE
            ocr_text = pytesseract.image_to_string(image, lang='ita', config='--psm 6')
            # st.write('color image ocr text')
            # st.write(ocr_text)
            # Split the text into lines
            lines = ocr_text.splitlines()
            # st.write(lines)
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
            # st.write(extracted_data_per_image_aliexpress)
            # st.write(data)
            df_extraction_aliexpress = pd.concat([df_extraction_aliexpress, extracted_data_per_image_aliexpress], ignore_index=True)        


        # Display the entire extracted text
        # st.subheader("Entire Extracted Text in Chinese")
        # st.write(ocr_text)

        
        # Create a DataFrame for the extracted data of this image
        df_extraction_image = pd.DataFrame([extracted_data_per_image])
        
        # st.write( 'df_extraction_image')
        # st.write(df_extraction_image)
        # Append the DataFrame for this image to the main df_extraction
        df_extraction = pd.concat([df_extraction, df_extraction_image], ignore_index=True)
    

    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'ALIEXPRESS']   
    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'TMALL'] 
    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'JD COM']   
    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'TAOBAO']   
    df_extraction = df_extraction[df_extraction['PLATFORM'] != 'CHINAALIBABA']   
  
    df_extraction_overall = pd.concat([df_extraction, df_extraction_aliexpress], ignore_index=True)
    df_extraction_overall = pd.concat([df_extraction_overall, df_extraction_tmall], ignore_index=True)
    df_extraction_overall = pd.concat([df_extraction_overall, df_extraction_jd], ignore_index=True)
    df_extraction_overall = pd.concat([df_extraction_overall, df_extraction_taobao], ignore_index=True)
    df_extraction_overall = pd.concat([df_extraction_overall, df_extraction_chinaalibaba], ignore_index=True)

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
        pass
    else:

        # # Check for missing values and replace them with an empty string
        tmall_df['企业注册号'].fillna('', inplace=True)
        tmall_df['SELLER_VAT_N'] = tmall_df['企业注册号']
        # Remove all white spaces in the '企业注册号' column
        tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].astype(str)
        tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
        tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.replace(r'，', '', regex=True)
        tmall_df['SELLER_VAT_N'] = tmall_df['SELLER_VAT_N'].str.strip()

        tmall_df['SELLER_BUSINESS_NAME_CN'] = tmall_df['企业名称']
      
        # Define a list of columns that may contain the company address
        address_columns = ['住所', '住 。所', '住 “所', '住 所']
        # Iterate through the rows and fill 'SELLER_ADDRESS_CN' with the first non-empty value from available columns
        for index, row in tmall_df.iterrows():
            for column in address_columns:
                if column in row and not pd.isna(row[column]) and row[column] != '':
                    tmall_df.at[index, 'SELLER_ADDRESS_CN'] = row[column]
                    break  # Stop looking for the address once found

        
        # Define a list of columns that may contain the company address
        type_columns = ['类型', '类 型', '类 ”型', '类 。 型']
        # Iterate through the rows and fill 'SELLER_ADDRESS_CN' with the first non-empty value from available columns
        for index, row in tmall_df.iterrows():
            for column in type_columns:
                if column in row and not pd.isna(row[column]) and row[column] != '':
                    tmall_df.at[index, 'COMPANY_TYPE_CN'] = row[column]
                    break  # Stop looking for the address once found

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
        # st.header(tmall_df)
        # st.dataframe(tmall_df)

# ------------------------------------------------------------
#                             TAOBAO
# ------------------------------------------------------------
   
    # Check if taobao_df is empty
    if taobao_df.empty:
        pass
    else:
        # Check for missing values and replace them with an empty string
        taobao_df['SELLER_VAT_N'] = taobao_df['统一社会信用代码 / 营业执照注册号']
        # taobao_df['SELLER_VAT_N'] = taobao_df['SELLER_VAT_N'].str.split('/').str[0]
        # Remove all white spaces in the '企业注册号' column
        # taobao_df['SELLER_VAT_N'] = taobao_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)

        taobao_df['SELLER_BUSINESS_NAME_CN'] = taobao_df['公司名称']
        # taobao_df['SELLER_BUSINESS_NAME_CN'] = taobao_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'\s', '', regex=True)

        taobao_df['COMPANY_TYPE_CN'] = taobao_df['公司类型']
        # taobao_df['COMPANY_TYPE_CN'] = taobao_df['COMPANY_TYPE'].str.replace(r'\s', '', regex=True)    

        taobao_df['SELLER_ADDRESS_CN'] = taobao_df['注册地址']
        taobao_df['SELLER_ADDRESS_CN'] = taobao_df['SELLER_ADDRESS_CN'].str.upper()
        # taobao_df['SELLER_ADDRESS_CN'] = taobao_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)    

        taobao_df['LEGAL_REPRESENTATIVE_CN'] = taobao_df['法定代表人']
        # taobao_df['LEGAL_REPRESENTATIVE_CN'] = taobao_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)    

        taobao_df['BUSINESS_DESCRIPTION'] = taobao_df['经营范围'].fillna('') + taobao_df['经营学围'].fillna('')

        taobao_df['ESTABLISHED_IN'] = taobao_df['经营期限自'].str.split('经').str[0]
        taobao_df['ESTABLISHED_IN'] = taobao_df['ESTABLISHED_IN'].str.replace(r'\s', '', regex=True)    

        taobao_df['INITIAL_CAPITAL'] = taobao_df['注册资本'].str.split('营').str[0]
        # taobao_df['INITIAL_CAPITAL'] = taobao_df['INITIAL_CAPITAL'].str.split('|').str[0]

        expiration_columns = ['经营期限至', "经营期限至'"]
        # Iterate through the rows and fill 'SELLER_BUSINESS_NAME_CN' with the first non-empty value from available columns
        for index, row in taobao_df.iterrows():
            for column in expiration_columns:
                if column in row and not pd.isna(row[column]) and row[column] != '':
                    taobao_df.at[index, 'EXPIRATION_DATE'] = row[column]
                    break  # Stop looking for the address once found

        taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['登记机关']
        # taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['REGISTRATION_INSTITUTION'].str.replace(r'\s', '', regex=True)        
        # st.dataframe(taobao_df)
    
# ------------------------------------------------------------
#                             1688
# ------------------------------------------------------------
    if chinaalibaba_df.empty:
        pass
    else:
        chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['公司名称']
        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['注册地址']
        chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['统一社会']

        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['企业类型']

        chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['法定代表人']        
        chinaalibaba_df['BUSINESS_DESCRIPTION'] = chinaalibaba_df['经营范围']
        chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['营业期限']

        chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['注册资本']
        chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['营业期限'].str.replace(r'^.*至今', 'Active', regex=True)

        chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['登记机关']
  
# ------------------------------------------------------------
#                             JD COM
# ------------------------------------------------------------

    if jd_df.empty:
        pass
    else:

        # # Check for missing values and replace them with an empty string
        jd_df['SELLER_VAT_N'] = jd_df['营业执照注册号'] 

        # Remove all white spaces in the '企业注册号' column
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].astype(str)
        # jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'，', '', regex=False)
        jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'。。', '', regex=False)
     
        # Define a list of columns that may contain the company name
        name_columns = ['企业名称', '企业和名称']
        # Iterate through the rows and fill 'SELLER_BUSINESS_NAME_CN' with the first non-empty value from available columns
        for index, row in jd_df.iterrows():
            for column in name_columns:
                if column in row and not pd.isna(row[column]) and row[column] != '':
                    jd_df.at[index, 'SELLER_BUSINESS_NAME_CN'] = row[column]
                    break  # Stop looking for the address once found
        

        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.lstrip()
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'CO\., LIMITED.*', 'CO., LIMITED', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'Co\., Limited.*', 'Co., Limited', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ED\.AR', 'ED', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ITEDAR', 'ITED', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ITEDAES', 'ITED', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'itedSWAR', 'ited', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ITEDAAEM', 'ITED', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('ITEDaA', 'ITED', regex=False)

        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'ARMSLIM').str[0]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'AES').str[0]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'ARMS').str[0]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ARES.*', '', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'aA.*', '', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'SUAEE.*', '', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'AARC.*', '', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'AES.*', '', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'AEM').str[0]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'CO.*LIMITED', 'CO., LIMITED', regex=True)

        jd_df['SELLER_ADDRESS_CITY'] = jd_df['营业执照所在地']
        jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.lstrip()
        jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.replace('1', '', regex=False)
       
        jd_df['SELLER_ADDRESS_CN'] = jd_df['联系地址']
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.lstrip()
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.upper()
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].fillna('-')

        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['法定代表人姓名']
        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].astype(str).str.lstrip()

        jd_df['SHOP_NAMEextracted'] = jd_df['店铺名称']

        if '店铺网址' in jd_df.columns:
            jd_df['SHOP_URLextracted'] = jd_df['店铺网址']
        else:
            jd_df['SHOP_URLextracted'] = '-'

        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.replace('Nmaljd', '//malljd', regex=False)
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.replace('maljd', '//malljd', regex=False)
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.replace(r'indexr', 'index-', regex=False)
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.replace(r'coryindex', '.com/index', regex=False)
        jd_df['SHOP_URLextracted'] = 'https://mall.jd.com/index-' + jd_df['SHOP_URLextracted'].str.split('index-').str[1].astype(str)
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.split('html').str[0] + 'html'
        jd_df['SHOP_URLextracted'] = jd_df['SHOP_URLextracted'].str.replace(r'^.*nanhtml', '-', regex=True)

        jd_df['BUSINESS_DESCRIPTION'] = jd_df['营业执照经营范围']

        jd_df['INITIAL_CAPITAL'] = jd_df['企业注册资金']
        # jd_df['INITIAL_CAPITAL'] = jd_df['INITIAL_CAPITAL'].str.split('|').str[0]

        jd_df['EXPIRATION_DATE'] = jd_df['营业执照有效期']
        # jd_df['EXPIRATION_DATE'] = jd_df['EXPIRATION_DATE'].str.split('|').str[0]
        jd_df['ESTABLISHED_IN'] = jd_df['EXPIRATION_DATE'].str.split('至').str[0]
        # jd_df['ESTABLISHED_IN'] = jd_df['ESTABLISHED_IN'].str.split('|').str[0]
        jd_df['ESTABLISHED_IN'] = jd_df['ESTABLISHED_IN'].str.lstrip()
        jd_df['EXPIRATION_DATE'] = jd_df['EXPIRATION_DATE'].str.split('至').str[1]

        jd_df = jd_df.replace(r'\。。', '', regex=True)
        jd_df = jd_df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)

        # jd_df = jd_df['SELLER_VAT_N'].fillna('-')

# ------------------------------------------------------------
#                             ALIEXPRESS
# ------------------------------------------------------------

    if aliexpress_df.empty:
        pass
    else:
        # try:
            # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['Nome della società']
        # except KeyError:
            # aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['Company name']
        
        
        # Define a list of columns that may contain the company name
        name_columns = ['Nome della società', 'Company name']
        # Iterate through the rows and fill 'SELLER_BUSINESS_NAME_CN' with the first non-empty value from available columns
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'SELLER_BUSINESS_NAME'] = row[column]
                        break
        except KeyError:
            aliexpress_df['SELLER_BUSINESS_NAME'] = '-'
            
        
        # Remove leading and trailing spaces
        aliexpress_df['SELLER_BUSINESS_NAME'] = aliexpress_df['SELLER_BUSINESS_NAME'].str.strip()

        aliexpress_df['COMPANY_TYPE'] = '-'


        # Define a list of columns that may contain the company name
        name_columns = ['Partita IVA', 'VAT number']
        for index, row in aliexpress_df.iterrows():
            for column in name_columns:
                if column in row and not pd.isna(row[column]) and row[column] != '':
                    aliexpress_df.at[index, 'SELLER_VAT_N'] = row[column]
                    break

        # try:
        #     aliexpress_df['SELLER_VAT_N'] = aliexpress_df['Partita.IVA']
        # except KeyError:
        #     aliexpress_df['SELLER_VAT_N'] = aliexpress_df['Partita IVA']
        # Remove 'Numero di' and the text before it if it's present
        aliexpress_df['SELLER_VAT_N'] = aliexpress_df['SELLER_VAT_N'].str.replace(r'Numero di.*$', '', regex=True)
        # aliexpress_df['SELLER_VAT_N'] = aliexpress_df['SELLER_VAT_N'].str.replace(r'registrazione.*$', '', regex=True)
        # Remove leading and trailing spaces
        aliexpress_df['SELLER_VAT_N'] = aliexpress_df['SELLER_VAT_N'].str.strip()

        # Define a list of columns that may contain the company name
        name_columns = ['Stabilito', 'Established']
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'ESTABLISHED_IN'] = row[column]
                        break
        except KeyError:
            aliexpress_df['ESTABLISHED_IN'] = '-'

        # try:
        #     aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['Stabilito']
        # except KeyError:
        #     aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['. Stabilito']
        
        aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.strip()
        aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.replace('Autorità di', '-', regex=False)
        aliexpress_df['REGISTRATION_INSTITUTION'] = aliexpress_df['ESTABLISHED_IN'].str.split(' - ').str[1]
        aliexpress_df['ESTABLISHED_IN'] = aliexpress_df['ESTABLISHED_IN'].str.split(' -').str[0]     

        name_columns = ['Indirizzo', 'Address']
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'SELLER_ADDRESS'] = row[column]
                        break
        except KeyError:
            aliexpress_df['SELLER_ADDRESS'] = '-'
            
        # aliexpress_df['SELLER_ADDRESS'] = aliexpress_df['Indirizzo']
   
        name_columns = ['E-mail', 'Email']
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'SELLER_EMAIL'] = row[column]
                        break
        except KeyError:
            aliexpress_df['SELLER_EMAIL'] = '-'

        
        aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.strip()
        aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.split(' ').str[0]  
        aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace('qg.com', 'qq.com', regex=False)
        aliexpress_df['SELLER_EMAIL'] = aliexpress_df['SELLER_EMAIL'].str.replace('qgq.com', 'qq.com', regex=False)

      
        name_columns = ['Numero di telefono', 'Phone Number']
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'SELLER_TEL_N'] = row[column]
                        break
        except KeyError:
            aliexpress_df['SELLER_TEL_N'] = '-'   
        
        aliexpress_df['SELLER_TEL_N'] = aliexpress_df['Numero di telefono']

        name_columns = ['Rappresentante legale', 'Legal Representative']
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'LEGAL_REPRESENTATIVE'] = row[column]
                        break
        except KeyError:
            aliexpress_df['LEGAL_REPRESENTATIVE'] = '-'   

        
        aliexpress_df['INITIAL_CAPITAL'] = '-'
        aliexpress_df['EXPIRATION_DATE'] = '-'

        name_columns = ['Business Scope', 'Ambito di attività']
        try:
            for index, row in aliexpress_df.iterrows():
                for column in name_columns:
                    if column in row and not pd.isna(row[column]) and row[column] != '':
                        aliexpress_df.at[index, 'BUSINESS_DESCRIPTION'] = row[column]
                        break
        except KeyError:
            aliexpress_df['BUSINESS_DESCRIPTION'] = '-'           
        

# ------------------------------------------------------------
#                             TEST - UNKNOWN PLATFORMS
# ------------------------------------------------------------

    if test_df.empty:
        pass
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


    # Concatenate them into a single DataFrame
    sellers_info_df = pd.concat([tmall_df, taobao_df, chinaalibaba_df, jd_df, aliexpress_df, test_df], ignore_index=True)
   # List of columns to add if they are missing
    columns_to_add = ['SELLER_BUSINESS_NAME', 'BUSINESS_DESCRIPTION', 'INITIAL_CAPITAL', 'EXPIRATION_DATE', 'SELLER_BUSINESS_NAME_CN', 'SELLER_ADDRESS', 'SELLER_ADDRESS_CN', 'COMPANY_TYPE', 'SELLER_EMAIL', 'SELLER_TEL_N', 'LEGAL_REPRESENTATIVE', 'LEGAL_REPRESENTATIVE_CN', 'REGISTRATION_INSTITUTION', 'COMPANY_TYPE_CN', 'SELLER_ADDRESS_CITY', 'SHOP_NAMEextracted', 'SHOP_URLextracted']

    # Check if columns are not in sellers_info_df and add them
    for column in columns_to_add:
        if column not in sellers_info_df.columns:
            sellers_info_df[column] = None  # You can replace None with the default value you want to use

    # Now sellers_info_df has 'SELLER_BUSINESS_NAME' and 'SELLER_ADDRESS' columns, or they were left as they were if they already existed
   
    # st.header('SELLER INFO-1')
    # st.write(sellers_info_df)



    sellers_info_df['SHOP_NAMEextracted'] = sellers_info_df['SHOP_NAMEextracted'].fillna('-')
    sellers_info_df['SHOP_URLextracted'] = sellers_info_df['SHOP_URLextracted'].fillna('-')


    def fill_empty_with_translation(df, target_column, source_column):
        for index, row in df.iterrows():
            if pd.isna(row[target_column]) and not pd.isna(row[source_column]):
                df.at[index, target_column] = row[source_column]

    fill_empty_with_translation(sellers_info_df, 'SELLER_BUSINESS_NAME', 'SELLER_BUSINESS_NAME_CN')


    # fill_empty_with_translation(sellers_info_df, 'SELLER_BUSINESS_NAME', 'SELLER_BUSINESS_NAME_CN')
    fill_empty_with_translation(sellers_info_df, 'SELLER_ADDRESS', 'SELLER_ADDRESS_CN')
    fill_empty_with_translation(sellers_info_df, 'COMPANY_TYPE', 'COMPANY_TYPE_CN')

    def format_personal_names(df, target_column, source_column):
        for index, row in df.iterrows():
            if pd.isna(row[target_column]) and not pd.isna(row[source_column]):
                if contains_chinese(row[source_column]):
                    try:
                        pinyin_text = lazy_pinyin(row[source_column])
                        formatted_name = format_chinese_name(pinyin_text).title()
                        df.at[index, target_column] = formatted_name
                    except Exception as e:
                        print(f"Formatting error: {e}")
                else:
                    df.at[index, target_column] = row[source_column]

    def contains_chinese(text):
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def format_chinese_name(pinyin_name):
        if len(pinyin_name) > 1:
            return pinyin_name[0] + ' ' + ''.join(pinyin_name[1:])
        else:
            return pinyin_name[0]
        
    format_personal_names(sellers_info_df, 'LEGAL_REPRESENTATIVE', 'LEGAL_REPRESENTATIVE_CN')
    # st.header('SELLERS INFO TRANSLATED')
    # st.write(sellers_info_df)



    # Sample Chinese addresses
    addresses = sellers_info_df['SELLER_ADDRESS_CN']


    # Function to extract city and province from an address using Jieba and the city_to_province dictionary
    def extract_city_and_province(address):
        if isinstance(address, str):  # Check if it's a string
            words = list(jieba.cut(address, cut_all=False))  # Segment the address into words
            # st.write(words)
            for word in words:
                if word.endswith('市'):
                    word = word[:-1]  # Remove the "市" character
                if word in city_to_province:
                    city = word
                    province = city_to_province[word]
                    return f'{city}市', province
        return "City not found", "Province not found"  # Return default values if city or province are not in the address

    # Create 'SELLER_CITY_CN' and 'SELLER_PROVINCE_CN' columns
    sellers_info_df['SELLER_CITY_CN'] = ""
    sellers_info_df['SELLER_PROVINCE_CN'] = ""

    # Process each address and assign the extracted city and province to the corresponding columns
    for index, address in enumerate(addresses):
        city, province = extract_city_and_province(address)
        sellers_info_df.at[index, 'SELLER_CITY_CN'] = city
        sellers_info_df.at[index, 'SELLER_PROVINCE_CN'] = province
    
    # Update 'SELLER_PROVINCE' and 'SELLER_CITY' if 'PLATFORM' is 'JD COM'
    condition = sellers_info_df['PLATFORM'] == 'JD COM'
    sellers_info_df.loc[condition & (sellers_info_df['SELLER_PROVINCE_CN'] == 'Province not found'), 'SELLER_PROVINCE_CN'] = sellers_info_df['SELLER_ADDRESS_CITY'].str.split('/').str[0]
    sellers_info_df.loc[condition & (sellers_info_df['SELLER_CITY_CN'] == 'City not found'), 'SELLER_CITY_CN'] = sellers_info_df['SELLER_ADDRESS_CITY'].str.split('/').str[1]
    
    provinces = sellers_info_df['SELLER_PROVINCE_CN']
    for index, province in enumerate(provinces):
        if province == '中国':
            sellers_info_df.at[index, 'SELLER_CITY_CN'] = '中国'

    # st.write(sellers_info_df)
    # Apply the translation function to the 'SELLER_ADDRESS' column
    sellers_info_df['SELLER_ADDRESS'] = sellers_info_df['SELLER_ADDRESS'].astype(str)
    # Apply these transformations based on the 'PLATFORM' value
    is_aliexpress = sellers_info_df['PLATFORM'] == 'ALIEXPRESS'
    # Apply transformations for 'ALIEXPRESS' platform
    sellers_info_df.loc[is_aliexpress, 'SELLER_ADDRESS'] = sellers_info_df.loc[is_aliexpress, 'SELLER_ADDRESS']
    # Apply transformations for other platforms
    sellers_info_df.loc[~is_aliexpress, 'SELLER_ADDRESS'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_ADDRESS_CN']
    sellers_info_df['SELLER_CITY'] = sellers_info_df['SELLER_CITY_CN']
    sellers_info_df['SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE_CN']
   
    # Sample DataFrame
    # Replace '-' in COMPANY_TYPE with 'Limited liability company' where COMPANY_NAME contains 'Co., Ltd'
    sellers_info_df.loc[sellers_info_df['COMPANY_TYPE'].isna() & sellers_info_df['SELLER_BUSINESS_NAME'].str.contains('Co., Ltd'), 'COMPANY_TYPE'] = 'Limited liability company'
    sellers_info_df.loc[(sellers_info_df['COMPANY_TYPE'] == r'-') & sellers_info_df['SELLER_BUSINESS_NAME'].str.contains('Trading Company'), 'COMPANY_TYPE'] = 'Trading Company'

    # Count the number of rows in sellers_info_df
    num_rows = sellers_info_df.shape[0]

    # Display the number of rows
    st.sidebar.subheader(f"{num_rows} seller(s) have been analysed")



    # Function to extract city, province, and country
    def extract_city_and_country(address):
        for city, province, country in country_city_dict:
            if city in address:
                return city, province, country
        return None, None, None

    provinces = sellers_info_df['SELLER_PROVINCE_CN']
    for index, province in enumerate(provinces):
        if province == '中国':
            sellers_info_df.at[index, 'SELLER_COUNTRY'] = 'Mainland China'
    sellers_info_df['SELLER_ADDRESS'] = sellers_info_df['SELLER_ADDRESS'].str.replace('Znen', 'Zhen',regex=False)
    # Apply the extraction function to each row
    sellers_info_df['SELLER_CITY'], sellers_info_df['SELLER_PROVINCE'], sellers_info_df['SELLER_COUNTRY'] = zip(*sellers_info_df['SELLER_ADDRESS'].apply(extract_city_and_country))
    mainland_china_rows = sellers_info_df[sellers_info_df['SELLER_COUNTRY'] == 'Mainland China']

    # Define a function to generate the URL based on 'SELLER_COUNTRY'
    def generate_aiqicha_url(row):
        if row['SELLER_COUNTRY'] == 'Mainland China':
            vat_number = str(row['SELLER_VAT_N'])  # Convert float to string          
            return 'https://www.aiqicha.com/s?q=' + vat_number
        else:
            return '-'
        

    # Apply the function to each row and update 'AIQICHA_URL' column
    sellers_info_df['AIQICHA_URL'] = sellers_info_df.apply(generate_aiqicha_url, axis=1)
    if uploaded_xlsx:
        sellers_info_df.update(df_urls)

    # Assuming sellers_info_df is your DataFrame
    sellers_info_df = sellers_info_df.applymap(lambda x: x if pd.notna(x) else "-")

    # This will replace NaN values with "-" in the sellers_info_df DataFrame    sellers_info_df = sellers_info_df.replace('', '-')

    sellers_info_df['SELLER_BUSINESS_NAME'] = sellers_info_df['SELLER_BUSINESS_NAME'].str.replace('Trade Bank', 'Wholesale Trading Company', regex=False)
    sellers_info_df['SELLER_BUSINESS_NAME'] = sellers_info_df['SELLER_BUSINESS_NAME'].str.replace('Trade Wholesale Bank', 'Wholesale Trading Company', regex=False)
    sellers_info_df['LEGAL_REPRESENTATIVE'] = sellers_info_df['LEGAL_REPRESENTATIVE'] + ' (' + sellers_info_df['LEGAL_REPRESENTATIVE_CN'] + ')'   
    sellers_info_df['COMPANY_LOCATION'] = sellers_info_df['SELLER_CITY'] + ', ' + sellers_info_df['SELLER_PROVINCE']
    sellers_info_df['COMPANY_LOCATION'] = sellers_info_df['COMPANY_LOCATION'].str.replace("City not found, Province Not Found", '-', regex=False)
    sellers_info_df['COMPANY_LOCATION'] = sellers_info_df['COMPANY_LOCATION'].str.replace("State, Province Not Found", '-', regex=False)
    sellers_info_df['SHOP_COMBINED'] = sellers_info_df['SELLER'] + '_' + sellers_info_df['PLATFORM']
    sellers_info_df['LEGAL_REPRESENTATIVE'] = sellers_info_df['LEGAL_REPRESENTATIVE'].str.replace("- (-)", '-', regex=False)
    sellers_info_df['LEGAL_REPRESENTATIVE'] = sellers_info_df['LEGAL_REPRESENTATIVE'].str.replace(" (-)", '', regex=False)
    sellers_info_df = sellers_info_df[['SHOP_NAMEextracted', 'SHOP_URLextracted', 'SELLER', 'SELLER_URL', "PLATFORM", "SHOP_COMBINED", "FILENAME", "SELLER_VAT_N", "SELLER_BUSINESS_NAME",  'SELLER_BUSINESS_NAME_CN', "COMPANY_TYPE", "SELLER_ADDRESS", 'SELLER_COUNTRY', 'COMPANY_LOCATION', 'SELLER_EMAIL', 'SELLER_TEL_N', "LEGAL_REPRESENTATIVE", "ESTABLISHED_IN", "INITIAL_CAPITAL", "EXPIRATION_DATE", 'AIQICHA_URL', "COMPANY_TYPE_CN", "SELLER_ADDRESS_CN", 'SELLER_PROVINCE', "SELLER_CITY", 'SELLER_PROVINCE_CN', 'SELLER_CITY_CN',  "LEGAL_REPRESENTATIVE_CN", "BUSINESS_DESCRIPTION",  "REGISTRATION_INSTITUTION"]]
    sellers_info_df['SELLER_VAT_N'] = sellers_info_df['SELLER_VAT_N'].str.strip()
    sellers_info_df['COMPANY_AREA'] = '-'
      
   

    urls = []
    locations = []
    countries = []
    areas = []
    
    # Iterate through the rows
    for index, row in sellers_info_df.iterrows():
        seller_vat_n = row['SELLER_VAT_N']

        # Check if 'SELLER_VAT_N' is an 8-digit number
        if len(str(seller_vat_n)) > 5 and len(str(seller_vat_n)) < 9:
            seller_name = row['SELLER_BUSINESS_NAME']
            seller_name = seller_name.lower()
            # Remove text after ' CO' if it exists
            if ' co' in seller_name:
                seller_name = seller_name.split(' co')[0]
            # Substitute whitespaces with '-'
            url = 'https://www.ltddir.com/companies/' + seller_name.replace(' ', '-') + '-co-limited/'
            urls.append(url)
            location = 'Hong Kong'
            locations.append(location)
            country = 'Hong Kong'
            countries.append(country)
        else:
            url = row['AIQICHA_URL']    
            urls.append(url)
            location = row['COMPANY_LOCATION'] 
            locations.append(location)
            area = row['COMPANY_AREA']
            areas.append(area)
            country = row['SELLER_COUNTRY']
            countries.append(country)

    sellers_info_df['AIQICHA_URL'] = urls
    sellers_info_df['COMPANY_LOCATION'] = locations
    sellers_info_df['SELLER_COUNTRY'] = countries



    # Define the previous regular expression pattern to match the desired element
    pattern = r'\b[A-Z0-9]{16,18}\b'       
    # Iterate through the rows and update the SELLER_VAT_N column
    for index, row in sellers_info_df.iterrows():
        vatn = row['SELLER_VAT_N']
        match = re.search(pattern, vatn)
        if match:
            element = match.group(0)
            sellers_info_df.at[index, 'SELLER_COUNTRY'] = 'Mainland China'

    def extract_area(country):
        if isinstance(country, str):  # Check if it's a string
            if country in country_area_dict:
                area = country_area_dict[country]
                return area
        return "-"  # Return default values if area is not in dictionary

    countries = sellers_info_df['SELLER_COUNTRY']
    # Process each address and assign the extracted city and province to the corresponding columns
    for index, country in enumerate(countries):
        area = extract_area(country)
        sellers_info_df.at[index, 'COMPANY_AREA'] = area


    output_df = pd.DataFrame
    output_df = sellers_info_df.copy()
    output_df['COMPANY_OFFICIAL_WEBSITE'] = '-' 
    output_df['COMPANY_STATUS'] = '-'
    output_df['COMPANY_ALERT_LEVEL'] = '-'
    output_df = output_df[['SELLER', 'SELLER_URL', "PLATFORM", "SHOP_COMBINED", "SELLER_BUSINESS_NAME", "SELLER_VAT_N", 'COMPANY_AREA', 'SELLER_COUNTRY', 'COMPANY_LOCATION', 'SELLER_ADDRESS', 'SELLER_BUSINESS_NAME_CN', 'SELLER_ADDRESS_CN', 'COMPANY_OFFICIAL_WEBSITE', 'SELLER_EMAIL', 'SELLER_TEL_N', 'LEGAL_REPRESENTATIVE', 'COMPANY_STATUS', 'COMPANY_ALERT_LEVEL', "ESTABLISHED_IN", 'AIQICHA_URL', "BUSINESS_DESCRIPTION"]]
    
    output_df = output_df.rename(columns={
    'SELLER': 'SHOP_NAME',
    'SELLER_URL': 'SHOP_URL',
    'SELLER_BUSINESS_NAME': 'BUSINESS_NAME',
    'SELLER_COUNTRY': 'COMPANY_COUNTRY',
    'SELLER_ADDRESS': 'COMPANY_ADDRESS',
    'SELLER_BUSINESS_NAME_CN': 'COMPANY_NAME_TRANSLATED',
    'SELLER_ADDRESS_CN': 'COMPANY_ADDRESS_TRANSLATED',
    'SELLER_EMAIL': 'COMPANY_EMAIL', 
    'SELLER_TEL_N': 'COMPANY_TEL_N', 
    'LEGAL_REPRESENTATIVE': 'COMPANY_CONTACT_PERSON',
    'AIQICHA_URL': 'COMPANIES_DATABASE_URL'
})
    
    # Replace newlines with an empty string in all DataFrame cells
    output_df = output_df.replace('\n', '', regex=True)

    # Generate a timestamp for the filename
    timestamp = generate_timestamp()
    filename = f"SellersInfo_SQL{timestamp}.xlsx"

    # Determine the user's home directory and the appropriate path separator
    current_platform = sys_platform.system()
    if current_platform == 'Windows':
        # Windows OS
        home_directory = os.path.expanduser("~")
        path_separator = '\\'
    else:
        # Linux or MacOS
        home_directory = os.path.expanduser("~")
        path_separator = '/'
    
    # Define the path to save the Excel file
    download_directory = os.path.join(home_directory, "Downloads")
    os.makedirs(download_directory, exist_ok=True)  # Create the directory if it doesn't exist
    
    download_path = os.path.join(download_directory, filename)
    
    # Export the DataFrame to Excel
    output_df.to_excel(download_path, index=False)
    st.header('Extracted Seller Information - SQL Template')
    
    # Provide the download link
    st.markdown(f"Download the SQL Template file: [SellersInfo_{timestamp}.xlsx]({download_path})")
    st.dataframe(output_df)

    st.markdown(f"Download the Excel file: [SellersInfo_{timestamp}.xlsx]({download_path})")
    st.dataframe(sellers_info_df)



