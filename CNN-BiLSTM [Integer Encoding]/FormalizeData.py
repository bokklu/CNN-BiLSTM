import xlsxwriter as xlsx

def formalize_data(X_train,y_train,X_test,y_test,total_no_seqs,total_dict_number,score,pred_results):
    workbook = xlsx.Workbook('ClassificationResults.xlsx')
    worksheet = workbook.add_worksheet('Sheet_One')

    # Excel formatting
    bold_underline = workbook.add_format({'bold': True, 'underline': True})
    header_format = workbook.add_format()
    header_format.set_font_size(24)

    worksheet.write(0, 0, 'Dataset Information:', header_format)
    worksheet.write(10, 0, 'Results:', header_format)
    worksheet.write(16, 0, 'Actual Classification Results of X_test:', header_format)
    worksheet.write(16, 9, 'Predicted Classification of X_test:', header_format)
    worksheet.write(2, 0, 'Nos of X_train:', bold_underline)
    worksheet.write(3, 0, 'Nos of y_train:', bold_underline)
    worksheet.write(4, 0, 'Nos of X_test:', bold_underline)
    worksheet.write(5, 0, 'Nos of y_test:', bold_underline)
    worksheet.write(6, 0, 'Total nos of records:', bold_underline)
    worksheet.write(7, 0, 'Class Labels:', bold_underline)
    worksheet.write(11, 0, 'Dictionary Capacity:', bold_underline)
    worksheet.write(12, 0, 'Nof of epochs:', bold_underline)
    worksheet.write(13, 0, 'Training Accuracy:', bold_underline)

    worksheet.write(2, 2, len(X_train))
    worksheet.write(3, 2, len(y_train))
    worksheet.write(4, 2, len(X_test))
    worksheet.write(5, 2, len(y_test))
    worksheet.write(6, 2, total_no_seqs)
    worksheet.write(7, 2, '0,1')
    worksheet.write(11, 2, total_dict_number + 1)
    worksheet.write(12, 2, '100')
    worksheet.write(13, 2, score * 100)

    # Writing Actual Test Results
    row = 18
    for test_seq in X_test:
        worksheet.write(row, 0, test_seq)
        row += 1
    row = 18
    for test_res in y_test:
        worksheet.write(row, 1, test_res)
        row += 1

    # Writing Prediction Test Results
    row = 18
    for test_seq in X_test:
        worksheet.write(row, 9, test_seq)
        row += 1
    row = 18
    for pred_res in pred_results:
        worksheet.write(row, 10, pred_res)
        row += 1

    #Confusion Matrix
    worksheet.write(16,17,'Confusion Matrix:', header_format)
    worksheet.write(18,18,'Predict: 0',bold_underline)
    worksheet.write(18,19,'Predict: 1',bold_underline)
    worksheet.write(19,17,'Actual: 0',bold_underline)
    worksheet.write(20,17,'Actual: 1',bold_underline)

    A0_P0 = 0
    A0_P1 = 0
    A1_P0 = 0
    A1_P1 = 0
    for i in range(len(y_test)):
        actual_value = y_test[i]
        predicted_value = pred_results[i]

        if actual_value == 0 and predicted_value == 0:
            A0_P0 += 1
        elif actual_value == 0 and predicted_value == 1:
            A0_P1 += 1
        elif actual_value == 1 and predicted_value == 0:
            A1_P0 += 1
        elif actual_value == 1 and predicted_value == 1:
            A1_P1 += 1

    worksheet.write(19,18,A0_P0)
    worksheet.write(19,19,A0_P1)
    worksheet.write(20,18,A1_P0)
    worksheet.write(20,19,A1_P1)
    worksheet.write(21,18,A0_P0+A1_P0)
    worksheet.write(21,19,A0_P1+A1_P1)
    worksheet.write(19,20,A0_P0+A0_P1)
    worksheet.write(20,20,A1_P0+A1_P1)

    workbook.close()