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
from geo_dict import city_to_province

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

            targets = targets_tmall
            
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

            targets = targets_taobao

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
            targets = targets_chinaalibaba

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
            # Assign the 'PLATFORM' and 'FILENAME' values in the extracted_data_per_image dictionary
            # updated_data = {}
            # for key, value in data.items():
            #     if '企业注册号' in key:
            #         key = '企业注册号'
            #     updated_data[key] = value
            # Replace the original data dictionary with the updated one
            # data = updated_data
            extracted_data_per_image_jd = pd.DataFrame([data])
            extracted_data_per_image_jd['PLATFORM'] = 'JD COM'
            extracted_data_per_image_jd['FILENAME'] = uploaded_image.name
            extracted_data_per_image_jd['FULL_TEXT'] = ocr_text
            # st.write(extracted_data_per_image_jd)
            # st.header('JD DATA EXTRACTED IN DICTIONARY')
            # st.write(data)
            df_extraction_jd = pd.concat([df_extraction_jd, extracted_data_per_image_jd], ignore_index=True)
            targets = targets_jd



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

            targets = targets_aliexpress
        

        if extracted_data_per_image['PLATFORM'] == 'TEST' or None:
            targets = targets_test

        # Display the entire extracted text
        # st.subheader("Entire Extracted Text in Chinese")
        # st.write(ocr_text)

        for word in targets:
            # Find the coordinates where the text is found
            target_text = word
            text_location = [(m.start(0), m.end(0)) for m in re.finditer(target_text, ocr_text)]

            if text_location:
                extracted_texts = ''
                if extracted_data_per_image['PLATFORM'] == 'TMALL' or 'JD COM' or 'ALIEXPRESS' or 'TAOBAO' or 'CHINAALIBABA':
                    # # Shift the coordinates to the right by 7 pixels
                    # roi_data = [(start + 4, end + 50) for start, end in text_location]
                    # extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                    pass

                elif extracted_data_per_image['PLATFORM'] == 'TEST':
                    # Shift the coordinates to the right by 7 pixels
                    roi_data = [(start + 0, end + 50) for start, end in text_location]
                    extracted_texts = [ocr_text[start:end] for start, end in roi_data]
                
                if extracted_texts:
                    # st.write(f"Extracted Text for '{target_text}' (Shifted by pixels to the right):")
                    # st.write(extracted_texts)
                    # Assign the extracted text to the dictionary
                    extracted_data_per_image[word] = extracted_texts
            else:
                # If target text is not found, assign an empty list
                extracted_data_per_image[word] = ['']
        
        # Create a DataFrame for the extracted data of this image
        df_extraction_image = pd.DataFrame(extracted_data_per_image)
        
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
   # st.header('df extraXTION')
    # st.write(df_extraction)
    # st.header('df extraXTION ALIEXPRESS')
    # st.write(df_extraction_aliexpress)
    # st.header('df extraXTION TMALL')
    # st.write(df_extraction_tmall)

    # st.header('df extraXTION OVERALL')
    # st.write(df_extraction_overall)

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
        # Create new columns based on the items in targets_taobao
        for target in targets_taobao:
            taobao_df[target] = None  # Add new columns with None values
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

        # taobao_df['EXPIRATION_DATE'] = taobao_df["经营期限至'"]
        # taobao_df['EXPIRATION_DATE'] = taobao_df['EXPIRATION_DATE'].str.replace(r'\s', '', regex=True)    

        taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['登记机关']
        # taobao_df['REGISTRATION_INSTITUTION'] = taobao_df['REGISTRATION_INSTITUTION'].str.replace(r'\s', '', regex=True)        
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
        
        # chinaalibaba_df['统一社会'].fillna('', inplace=True)
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['FULL_TEXT'].str.split('注册号').str[1]
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.split('企业').str[0]
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.split('登记').str[0]
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.replace(r'\s', '', regex=True)
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].astype(str)
        # chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['SELLER_VAT_N'].str.split('企业').str[0]
       
        # chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['公司名称'].str.split('注').str[0]
        chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['公司名称']
        # chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['SELLER_BUSINESS_NAME_CN'].str.strip()
        # chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['SELLER_BUSINESS_NAME_CN'].str.split('认证').str[1]
        chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['注册地址']
        chinaalibaba_df['SELLER_VAT_N'] = chinaalibaba_df['统一社会']

        # # Define the previous regular expression pattern to match the desired element
        # pattern = r'\b[A-Z0-9]{16,18}\b'       
        # # Iterate through the rows and update the SELLER_VAT_N column
        # for index, row in chinaalibaba_df.iterrows():
        #     text = row['FULL_TEXT']
        #     match = re.search(pattern, text)
        #     if match:
        #         element = match.group(0)
        #         chinaalibaba_df.at[index, 'SELLER_VAT_N'] = element

        # chinaalibaba_df['SELLER_BUSINESS_NAME_CN'] = chinaalibaba_df['SELLER_BUSINESS_NAME_CN'].str.split(' ').str[0]
        # # 主体资质 ”该信息于2023年06月04日通过联信专业认证  深圳市道勤酒店用品有限公司                          中国广东深圳龙岗区南湾街道布澜路21号 联创科技园25号厂房3楼301-1  5000万元 2009-07-28  91440300692529553R 芭国龙  91440300692529553R                            有限责任公司  深圳市市场监督管理局                             2023  2009-07-27 至 至今 
        chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['企业类型']
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.split('登记').str[0]
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.split('上册').str[0]
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'主', '', regex=True)
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'串', '', regex=True)
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'估', '', regex=True)
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'型', '', regex=True)
        # chinaalibaba_df['COMPANY_TYPE_CN'] = chinaalibaba_df['COMPANY_TYPE_CN'].str.replace(r'\s', '', regex=True)

        # chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['地址'].str.split('成立').str[0]
        # chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.split('|').str[0]
        # chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.split('注册').str[0]
        # chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)
        # chinaalibaba_df['SELLER_ADDRESS_CN'] = chinaalibaba_df['SELLER_ADDRESS_CN'].str.upper()

        chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['法定代表人']        
        # chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'].str.split('表人').str[1]
        # chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'].str.split('注册').str[0]
        # chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'] = chinaalibaba_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)

        chinaalibaba_df['BUSINESS_DESCRIPTION'] = chinaalibaba_df['经营范围']
        chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['营业期限']

        # chinaalibaba_df['ESTABLISHED_IN'] = chinaalibaba_df['注册资本'].str.split('成立日期').str[1]
        
        # # Define a regular expression pattern to match the "4 digits - 2 digits - 2 digits" pattern
        # date_pattern = r'(\d{4}-\d{2}-\d{2})'
        # # Iterate through the rows and update the ESTABLISHED IN column
        # for index, row in chinaalibaba_df.iterrows():
        #     text = row['FULL_TEXT']
        #     match = re.search(date_pattern, text)
        #     if match:
        #         date_string = match.group(1)  # Extract the first matching date
        #         chinaalibaba_df.at[index, 'ESTABLISHED_IN'] = date_string

        chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['注册资本']
        # chinaalibaba_df['INITIAL_CAPITAL'] = chinaalibaba_df['INITIAL_CAPITAL'].str.split('|').str[0]

        # chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['营业期限'].str.split('经').str[0]
        # chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['EXPIRATION_DATE'].str.split('|').str[0]
        chinaalibaba_df['EXPIRATION_DATE'] = chinaalibaba_df['营业期限'].str.replace(r'^.*至今', 'Active', regex=True)

        chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['登记机关']
        # chinaalibaba_df['REGISTRATION_INSTITUTION'] = chinaalibaba_df['REGISTRATION_INSTITUTION'].str.split('|').str[0]
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
        jd_df['SELLER_VAT_N'] = jd_df['营业执照注册号'] 
        # jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.split('丢定代').str[0]
        # jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'法证代', '法定代', regex=True)
        # jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.split('法定代').str[0]
        # jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.replace(r'\。', '', regex=True)
        # jd_df['SELLER_VAT_N'] = jd_df['SELLER_VAT_N'].str.split(r'|').str[0]

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
        
        # if '企业和名称' in jd_df.columns:
        #     jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['企业名称'].fillna(jd_df['企业和名称'])
        # else: 
        #     jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['企业名称']

        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.lstrip()
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'\:').str[1]
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'CO\., LIMITED.*', 'CO., LIMITED', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'Co\., Limited.*', 'Co., Limited', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ED\.AR', 'ED', regex=True)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ITEDAR', 'ITED', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ITEDAES', 'ITED', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'itedSWAR', 'ited', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace(r'ITEDAAEM', 'ITED', regex=False)
        jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('ITEDaA', 'ITED', regex=False)

        # # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'Limited').str[0] + 'Limited'
        # # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split(r'Company').str[0] + 'Company'
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

        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('UMITED', ' LIMITED', regex=False)

        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('UMITED', ' LIMITED', regex=False)
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('co. UMITED', 'CO., LIMITED', regex=False)
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('CO .LIMITED', 'CO., LIMITED', regex=False)
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('CO\\.LIMITED', 'CO., LIMITED', regex=True)
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('UIMITED', ' LIMITED', regex=False)
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.replace('Intemational', 'International', regex=False)
        # Check if 'CO., LIMITED' is not found in 'SELLERBUSINESSNAMECN'
        # mask = ~jd_df['SELLER_BUSINESS_NAME_CN'].str.contains('CO\\., LIMITED', case=False, na=False)
        # # Replace 'LIMITED' with 'CO., LIMITED' for rows where it's not found
        # jd_df.loc[mask, 'SELLER_BUSINESS_NAME_CN'] = jd_df.loc[mask, 'SELLER_BUSINESS_NAME_CN'].str.replace('LIMITED', 'CO., LIMITED', regex=False)
     # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split('有限公司').str[0] + '有限公司'
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.split('综合').str[0]
        # jd_df['SELLER_BUSINESS_NAME_CN'] = jd_df['SELLER_BUSINESS_NAME_CN'].str.lstrip()

        jd_df['SELLER_ADDRESS_CITY'] = jd_df['营业执照所在地']
        jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.lstrip()
        jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.replace('1', '', regex=False)
        # jd_df['SELLER_ADDRESS_CITY'] = jd_df['SELLER_ADDRESS_CITY'].str.replace(r'\。', '', regex=True)
       
        jd_df['SELLER_ADDRESS_CN'] = jd_df['联系地址']
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.lstrip()
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.upper()
        jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].fillna('-')
       # jd_df['SELLER_ADDRESS_CN'] = jd_df['SELLER_ADDRESS_CN'].str.replace(r'\s', '', regex=True)

        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['法定代表人姓名']
        jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].astype(str).str.lstrip()
        # jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.slice(0, 3)

        # jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'\s', '', regex=True)
        # jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'人:', '', regex=False)
        # jd_df['LEGAL_REPRESENTATIVE_CN'] = jd_df['LEGAL_REPRESENTATIVE_CN'].str.replace(r'人;', '', regex=False)

        jd_df['SHOP_NAMEextracted'] = jd_df['店铺名称']
        # jd_df['SHOP_NAMEextracted'] = jd_df['SHOP_NAMEextracted'].str.replace('店生网址', '店铺网址', regex=False)
        # jd_df['SHOP_NAMEextracted'] = jd_df['SHOP_NAMEextracted'].str.split('店铺网').str[0]
        # jd_df['SHOP_NAMEextracted'] = jd_df['SHOP_NAMEextracted'].str.lstrip()

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

        aliexpress_df['INITIAL_CAPITAL'] = '-'
        aliexpress_df['EXPIRATION_DATE'] = '-'
        aliexpress_df['BUSINESS_DESCRIPTION'] = '-'


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
    # sellers_info_df.drop(['统一社会', '企业注册号', '注册号', '公司名称', '企业名称', '企业类型', '地址', '成立日期', '注册号', '类型', '类 型', '类 ”型', '类 。 型', '地址', '住所', '住 所', '住 ”所', '法定代表人', '经营期限自', '经营范围', '经营学围', '成立时间', '注册资本', '营业期限', '登记机关'], axis=1, inplace=True)
    # sellers_info_df.drop(['该准时间'], axis=1, inplace=True)
    # Apply the translation function to the 'SELLER_BUSINESS_NAME_CN' column
    # sellers_info_df['SELLER_BUSINESS_NAME'] = sellers_info_df['SELLER_BUSINESS_NAME'].astype(str)
    # # Apply these transformations based on the 'PLATFORM' value
    # is_aliexpress = sellers_info_df['PLATFORM'] == 'ALIEXPRESS'
    # Apply transformations for 'ALIEXPRESS' platform
    # sellers_info_df.loc[is_aliexpress, 'SELLER_BUSINESS_NAME'] = sellers_info_df.loc[is_aliexpress, 'SELLER_BUSINESS_NAME']
    # Apply transformations for other platforms
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'].str.replace(r'贸易批发商行', 'Wholesale Trading Company', regex=True)
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN'].str.replace(r'商行', 'Trading Company', regex=True)
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME_CN']
    # sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME'] = sellers_info_df.loc[~is_aliexpress, 'SELLER_BUSINESS_NAME'].str.replace(r'Trade Trading Company', 'Trading Company', regex=True)
    # sellers_info_df.loc[~is_aliexpress, 'COMPANY_TYPE_EN'] = sellers_info_df.loc[~is_aliexpress, 'COMPANY_TYPE']
    # sellers_info_df.loc[~is_aliexpress, 'LEGAL_REPRESENTATIVE_EN'] = sellers_info_df.loc[~is_aliexpress, 'LEGAL_REPRESENTATIVE']



    def fill_empty_with_translation(df, target_column, source_column):
        for index, row in df.iterrows():
            if pd.isna(row[target_column]) and not pd.isna(row[source_column]):
                df.at[index, target_column] = row[source_column]

    fill_empty_with_translation(sellers_info_df, 'SELLER_BUSINESS_NAME', 'SELLER_BUSINESS_NAME_CN')


    # # Create a Translator instance
    # translator = Translator()

    # def fill_empty_with_translation(df, target_column, source_column):
    #     for index, row in df.iterrows():
    #         if pd.isna(row[target_column]) and not pd.isna(row[source_column]):
    #             try:
    #                 translation = translator.translate(row[source_column], src='zh-cn', dest='en')
    #                 df.at[index, target_column] = translation.text
    #             except Exception as e:
    #                 print(f"Translation error: {e}")

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


    # # Function to extract city from an address using Jieba
    # def extract_city(address):
    #     if isinstance(address, str):  # Check if it's a string
    #         words = list(jieba.cut(address, cut_all=False))  # Segment the address into words
    #         for word in words:
    #             if word.endswith('市'):
    #                 return word
    #             elif word.endswith('州'):   
    #                 return word + '市'
    #             elif word.endswith('圳'):
    #                 return word + '市'
    #     return "City not found" # Return a default value if the city is not in the address
    # # Process each address
    # # Create an empty 'SELLER_CITY' column
    # sellers_info_df['SELLER_CITY_CN'] = ""

    # # Process each address and assign the extracted city to the 'SELLER_CITY' column
    # for index, address in enumerate(addresses):
    #     city = extract_city(address)
    #     sellers_info_df.at[index, 'SELLER_CITY_CN'] = city
    



    # # Function to extract province from the city
    # def extract_province(city):
    #     if city in city_to_province:
    #         return city_to_province[city]
    #     return "Province not found"  # Default value if the city is not in the mapping

    # # Apply the extract_province function to the 'SELLER_CITY' column
    # sellers_info_df['SELLER_PROVINCE_CN'] = sellers_info_df['SELLER_CITY_CN'].apply(extract_province)

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


    country_city_dict = {
        ('Shenzhen', 'Guangdong', 'Mainland China'),
        ('Guangzhou', 'Guangdong', 'Mainland China'),
        ('Foshan', 'Guangdong', 'Mainland China'),
        ('Bengbu', 'Anhui', 'Mainland China'),
        ('Hefei', 'Anhui', 'Mainland China'),
        ('Nanning', 'Guangxi', 'Mainland China'),
        ('Shanghai', 'Shanghai', 'Mainland China'),
        ('Suqian', 'Jiangsu', 'Mainland China'),
        ('Langfang', 'Hebei', 'Mainland China'),
        ('China', 'China', 'Mainland China'),
        ('Zhangjiagang', 'Suzhou', 'Mainland China'),
        ('Yiwu', 'Zhejiang', 'Mainland China'),
        ('Lianyungang', 'Jiangsu', 'Mainland China'),
        ('Nanjing', 'Jiangsu', 'Mainland China'),
        ('Ningbo', 'Zhejiang', 'Mainland China'),
        ('Jinan', 'Shandong', 'Mainland China'),
        ('Nanchang', 'Jiangxi', 'Mainland China'),
        ('Yiwu', 'Zhejiang', 'Mainland China'),
        ('Beijing', 'Hebei', 'Mainland China'),
        ('Foshan', 'Guangdong', 'Mainland China'),
        ("Xi'an", 'Shaanxi', 'Mainland China'),
        ('Wuhu', 'Anhui', 'Mainland China'),
        ('Lianyungang', 'Jiangsu', 'Mainland China'),
        ('Changchun', 'Jilin', 'Mainland China'),
        ('Jinhua', 'Zhejiang', 'Mainland China'),
        ('Changsha', 'Hunan', 'Mainland China'),
        ('Guangyang', 'Henan', 'Mainland China'),
        ('Xinyu', 'Jiangxi', 'Mainland China'),
        ('Yidu', 'Hubei', 'Mainland China'),
        ('Lanzhou', 'Jiangsu', 'Mainland China'),
        ('Zhuozhuo', 'Hebei', 'Mainland China'),
        ('Taizhou', 'Zhejiang', 'Mainland China'),
        ('Gaozhou', 'Guangdong', 'Mainland China'),
        ('Changzhou', 'Jiangsu', 'Mainland China'),
        ('Quanzhou', 'Fujian', 'Mainland China'),
        ('Huzhou', 'Zhejiang', 'Mainland China'),
        ('Suzhou', 'Jiangsu', 'Mainland China'),
        ('Hangzhou', 'Zhejiang', 'Mainland China'),
        ('Dazhou', 'Hebei', 'Mainland China'),
        ('Dongguan', 'Guangdong', 'Mainland China'),
        ('Jiangmen', 'Guangdong', 'Mainland China'),
        ('Hengyang', 'Hunan', 'Mainland China'),
        ('Nantong', 'Jiangsu', 'Mainland China'),
        ('Dazhou', 'Hebei', 'Mainland China'),
        ('Luoyang', 'Henan', 'Mainland China'),
        ('Huangshan', 'Anhui', 'Mainland China'),
        ('Zhoukou', 'Henan', 'Mainland China'),
        ('Nanping', 'Fujian', 'Mainland China'),
        ('Jizhou', 'Henan', 'Mainland China'),
        ('Dongxiang', 'Jiangxi', 'Mainland China'),
        ('Guigang', 'Guangxi', 'Mainland China')
    }

    country_area_dict = {
        'Mainland China': 'Greater China',
        'Hong Kong': 'Greater China'
    }
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

    # Apply the extraction function to each row
    sellers_info_df['SELLER_CITY_'], sellers_info_df['SELLER_PROVINCE_'], sellers_info_df['SELLER_COUNTRY'] = zip(*sellers_info_df['SELLER_ADDRESS'].apply(extract_city_and_country))
    # Update 'SELLER_PROVINCE' if 'Province Not Found' and 'SELLER_PROVINCE_' is not None
    sellers_info_df.loc[(sellers_info_df['SELLER_PROVINCE'] == 'Province Not Found') & (sellers_info_df['SELLER_PROVINCE_'].notna()), 'SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE_']

    # Update 'SELLER_CITY' if 'City Not Found' and 'SELLER_CITY_' is not None
    sellers_info_df.loc[(sellers_info_df['SELLER_CITY'] == 'City not found') & (sellers_info_df['SELLER_CITY_'].notna()), 'SELLER_CITY'] = sellers_info_df['SELLER_CITY_']
    sellers_info_df['SELLER_CITY'] = sellers_info_df['SELLER_CITY'].str.replace(' City', '', regex=False)
    sellers_info_df['SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE'].str.replace(' Province', '', regex=False)
    sellers_info_df['SELLER_PROVINCE'] = sellers_info_df['SELLER_PROVINCE'].fillna(sellers_info_df['SELLER_PROVINCE_'])
    # sellers_info_df['SELLER_PROVINCE_'] = sellers_info_df['SELLER_PROVINCE_'].fillna(sellers_info_df['SELLER_PROVINCE'])
    sellers_info_df['SELLER_CITY'] = sellers_info_df['SELLER_CITY'].fillna(sellers_info_df['SELLER_CITY_'])
    # sellers_info_df['SELLER_CITY_'] = sellers_info_df['SELLER_CITY_'].fillna(sellers_info_df['SELLER_CITY'])
    # Apply the extraction function to each row
    sellers_info_df['SELLER_CITY_'], sellers_info_df['SELLER_PROVINCE_'], sellers_info_df['SELLER_COUNTRY'] = zip(*sellers_info_df['SELLER_CITY'].apply(extract_city_and_country))
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
    
    # Define the path to save the Excel file
    ## download_path = os.path.join("/Users/mirkofontana/Downloads", filename)

    # Export the DataFrame to Excel
    ## output_df.to_excel(download_path, index=False)
    ## st.header('Extracted Seller Information - SQL Template')
    # Provide the download link
    ## st.markdown(f"Download the SQL Template file: [SellersInfo_{timestamp}.xlsx]({download_path})")
    ## st.dataframe(output_df)
    

    # Determine the user's home directory and the appropriate path separator
    if platform.system() == 'Windows':
        # Windows OS
        home_directory = os.path.expanduser("~")
        path_separator = '\\'
    else:
        # Linux or MacOS
        home_directory = os.path.expanduser("~")
        path_separator = '/'
    
    # Define the path to save the Excel file
    download_path = os.path.join(home_directory, "Downloads", filename)
    
    # Export the DataFrame to Excel
    output_df.to_excel(download_path, index=False)
    st.header('Extracted Seller Information - SQL Template')
    
    # Provide the download link
    st.markdown(f"Download the SQL Template file: [SellersInfo_{timestamp}.xlsx]({download_path})")
    st.dataframe(output_df)
 
    # # Generate a timestamp for the filename
    # timestamp = generate_timestamp()
    # filename = f"SellersInfo_{timestamp}.xlsx"
    
    # # Define the path to save the Excel file
    # download_path = os.path.join("/Users/mirkofontana/Downloads", filename)

    # # Export the DataFrame to Excel
    # sellers_info_df.to_excel(download_path, index=False)
    # st.header('Extracted Seller Information')
    # # Provide the download link
    st.markdown(f"Download the Excel file: [SellersInfo_{timestamp}.xlsx]({download_path})")
    st.dataframe(sellers_info_df)



