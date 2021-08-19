import subprocess
path_data = '/hdd/hanoch/runmodels/img_quality/results/inference_production/train/softmap'
target_path = '/hdd/hanoch/runmodels/img_quality/results/inference_production/train/softmap/cls_3_low_avg_pooling_score'
g = {2312: '719ec169-edc6-5217-875c-8d5a03485426', 1123: '4ff45e7f-d4ff-5aa7-926f-94a972307c25', 744: 'a2576401-fe77-5b00-95f1-2b1c561ad013', 609: '2b7edf42-1389-5dbc-85c3-671caaefe15b', 773: 'faab809f-59d1-56c0-bb87-1818c560f29d', 2201: '8d79bf88-5702-55e3-bcbf-2da396a27631', 2023: 'a9d6bc86-e6e0-54ce-8187-26938ceb2460', 499: '5e72465d-768f-55c8-ab61-e2f62c2ada9b', 1217: '8bfd973f-f3c3-5ce9-a9ac-422ecda82603', 2047: '3b94b297-c85e-527c-af2d-68c1b0daafbe', 1887: 'e7bb59c1-dd60-5068-a882-d3254f9083ca', 2286: 'b315230a-92d7-527c-ac71-4db470503fc8', 1642: '73798fbb-1ec7-5670-aaf5-ef6b23b1d249', 2168: '3bbea3e0-a199-5ea6-8b87-af64fbca2dcd', 31: 'c148829a-a064-55ce-b7b5-a0e7d9b53ad4', 1296: 'b6722802-5155-5aa6-80c0-67b248e45ef0', 1247: 'ff8085fd-432d-512e-9d0f-fdf3ec963124', 1625: 'b5882e62-dccc-58f6-874f-f7da4a775802', 945: 'f44a2c7c-1446-55cb-b923-c1bcd31c5c35', 2107: '6e824de4-3bf3-578e-a392-c2e66534f0bd', 244: '2f8ac6b1-3f1e-5431-8397-942ed5984d99', 232: 'a54f199f-91cc-5cc0-953c-4f8a3a2d6627', 728: '0857df01-4425-5e27-888b-6fe69585d1a1', 659: '5e140bfc-64e3-545f-9d63-22e870d1902d', 2202: 'f1ac4cc7-61fb-561c-9123-b840453d4755', 1218: 'b83cfc61-23b3-52b7-aaef-56323fc6f3ee', 905: 'c921ec05-22aa-5c12-af99-e8999a5ea53a', 1719: '185a4b0c-96e9-5fca-ab18-1d1fcf59f0e3', 2374: 'b674a1ee-8d75-539f-9c69-4420ecb97911', 1175: '543a16a7-ba9c-5316-8ae8-1b2a5b44fb24'}

for p in g.keys():
    file_full_path = subprocess.getoutput('find ' + path_data + ' -iname ' + '"*' +g[p] +'*"')
    print(file_full_path)
    ans2 = subprocess.getoutput('cp -p ' +  file_full_path + ' ' + ' ' + target_path)

