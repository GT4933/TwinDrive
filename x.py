import os
import cv2
import numpy as np

wiped_files = [
    "bdd100k_00495359-1d04dd8a.png", "bdd100k_011d593a-89158b9d.png", "bdd100k_01c4035b-bcaeb067.png",
    "bdd100k_02133057-fc70cc0e.png", "bdd100k_03dda8e7-dea74947.png", "bdd100k_04629c39-1d564d8a.png",
    "bdd100k_049dde52-19569c8c.png", "bdd100k_056c5701-175a265c.png", "bdd100k_06d937fd-3df1b5be.png",
    "bdd100k_082d5dee-53c30dad.png", "bdd100k_0a493f24-352f747e.png", "bdd100k_0b2867ef-a6ca437b.png",
    "bdd100k_0d13ae3d-f033a349.png", "bdd100k_0d5b19b3-df617a03.png", "bdd100k_0efe0129-c499802e.png",
    "bdd100k_0fa91e8c-e1e98e63.png", "bdd100k_0fc07ccd-14991933.png", "bdd100k_0fe831f2-d3180400.png",
    "bdd100k_13980959-e2ec7280.png", "bdd100k_157a9cee-5ce9ec62.png", "bdd100k_15a72340-98d8dca5.png",
    "bdd100k_167e8187-7ea0424e.png", "bdd100k_16dce9f0-8ef6f581.png", "bdd100k_186ca5af-bbb863e1.png",
    "bdd100k_18a650d6-c4e34bc5.png", "bdd100k_18b57f39-4f41ecba.png", "bdd100k_1a2f950b-6289c313.png",
    "bdd100k_1b883059-144b2678.png", "bdd100k_1c1de1e7-346d6219.png", "bdd100k_1c29669f-1237cc57.png",
    "bdd100k_1c5a5eec-b5894678.png", "bdd100k_1cbc711c-205586b2.png", "bdd100k_1d0a7787-8bf8ca8a.png",
    "bdd100k_1d36dcb4-e4b70050.png", "bdd100k_1df69db3-64e475fd.png", "bdd100k_1e45b3c5-77f120fd.png",
    "bdd100k_1f77fb7b-d6304b0f.png", "bdd100k_1ff90038-ff3d047b.png", "bdd100k_20a8f0f8-2bd78b81.png",
    "bdd100k_2105fcd5-412eacce.png", "bdd100k_212289f5-732658d4.png", "bdd100k_21575924-6dcb9f33.png",
    "bdd100k_215b7b34-864de34f.png", "bdd100k_21649da7-18621e54.png", "bdd100k_219b6f67-0c151840.png",
    "bdd100k_22a14014-5c0db650.png", "bdd100k_22ca8f3f-1fdcbbce.png", "bdd100k_22d2864f-1e4f2110.png",
    "bdd100k_23445ef6-af90b81d.png", "bdd100k_23d675e1-a56f88ec.png", "bdd100k_24bff1fe-5824eac7.png",
    "bdd100k_2656e02e-787c933d.png", "bdd100k_26ed798f-e07dd4b6.png", "bdd100k_27248c22-a748e30f.png",
    "bdd100k_280c2a04-58450d9a.png", "bdd100k_2a051b12-a7404163.png", "bdd100k_2ae5baeb-6c3992ac.png",
    "bdd100k_2d89f625-5e44c56a.png", "bdd100k_2e5ae824-55520bfb.png", "bdd100k_2e8bd85f-7dda324c.png",
    "bdd100k_2ec0b931-cbad220e.png", "bdd100k_2ef8d318-ff4a81c4.png", "bdd100k_2f1b8110-37e9fbad.png",
    "bdd100k_301e99f5-f1a9d13c.png", "bdd100k_307ce959-b55ab09a.png", "bdd100k_30d60151-c2618b59.png",
    "bdd100k_31081723-b51998d4.png", "bdd100k_3170aba7-d3105b12.png", "bdd100k_31b7487a-a01d6351.png",
    "bdd100k_325bf3b3-7f830f7e.png", "bdd100k_32654172-8c1f23e8.png", "bdd100k_32cf0c44-faf48809.png",
    "bdd100k_340f5f82-892b6242.png", "bdd100k_34bd8b4d-00530890.png", "bdd100k_360cfc03-57f966fc.png",
    "bdd100k_36263df4-d49f42cd.png", "bdd100k_36ae0153-0b30a49b.png", "bdd100k_36b03591-64915045.png",
    "bdd100k_36c9fa2e-fa9be3aa.png", "bdd100k_382e37b6-139b8d9f.png", "bdd100k_38948b5b-b9872a87.png",
    "bdd100k_3ba04309-9cb1b58c.png", "bdd100k_3c08ec50-1d212cdf.png", "bdd100k_3dab6efd-ecb3083b.png",
    "bdd100k_3e0ad77d-66c1b120.png", "bdd100k_3f291ac3-d2abcdbd.png", "bdd100k_409b367f-e28dbcd1.png",
    "bdd100k_41895080-b4849245.png", "bdd100k_41ab28a6-dd5b84f7.png", "bdd100k_4283e5c3-5cf933f4.png",
    "bdd100k_443e0347-d3765366.png", "bdd100k_44c8ef75-37dd9bc2.png", "bdd100k_4510f933-8d77eacb.png",
    "bdd100k_4655bcd1-42baf9cb.png", "bdd100k_4b07927b-d7e2fecc.png", "bdd100k_4c3bf807-021593bc.png",
    "bdd100k_4c840aee-0dd28451.png", "bdd100k_4d733ba1-6ad4b922.png", "bdd100k_4d7b4b90-9bd3f150.png",
    "bdd100k_4d97e3fb-a9cf0e36.png", "bdd100k_4e87979c-ff4e3425.png", "bdd100k_4ec1bd82-814b3df6.png",
    "bdd100k_4faabd52-4948c2aa.png", "bdd100k_50c11f3a-16606cc7.png", "bdd100k_50ec43cf-e41192f7.png",
    "bdd100k_51a79f7c-9389f699.png", "bdd100k_51fff027-0da11077.png", "bdd100k_5269b2c4-ecc0610c.png",
    "bdd100k_54a92bc0-c0771ce8.png", "bdd100k_5861c948-a40cae6f.png", "bdd100k_5a815a91-ee140052.png",
    "bdd100k_5b9fc8b8-d1282781.png", "bdd100k_5daa9274-a4a24449.png", "bdd100k_5dbc01d7-b11a959b.png",
    "bdd100k_5dc4a461-50d0525a.png", "bdd100k_5e4b3dc5-d4b04e70.png", "bdd100k_5f178acc-8b810b49.png",
    "bdd100k_621fb031-4cb67511.png", "bdd100k_621fb031-8e212444.png", "bdd100k_63a626a0-d5485f8e.png",
    "bdd100k_64fdb05b-ed6d85c4.png", "bdd100k_65939f30-fc872d46.png", "bdd100k_6848adce-1a36c1d5.png",
    "bdd100k_68be5b8f-21c2824b.png", "bdd100k_68eae8d4-3ed5f45a.png", "bdd100k_696a7656-d72a16f0.png",
    "bdd100k_69d6744a-d22e5d94.png", "bdd100k_6a00c92c-09d0090e.png", "bdd100k_6a286fbc-ed149abf.png",
    "bdd100k_6a729de1-39eed982.png", "bdd100k_6ac8c7f6-130ffaa0.png", "bdd100k_6bacf712-3ebf051c.png",
    "bdd100k_6c173cf1-4335c0a4.png", "bdd100k_6c36756d-15d9310b.png", "bdd100k_6c6ea5d7-2bea120a.png",
    "bdd100k_6cf7a093-0748b3cc.png", "bdd100k_6d0d7bef-b4cc0870.png", "bdd100k_6e04c802-452bf70f.png",
    "bdd100k_6ea9d065-03305dcc.png", "bdd100k_705c6dce-697b096c.png", "bdd100k_70c1fd95-902e1f4a.png",
    "bdd100k_70d7553c-7da414da.png", "bdd100k_7142ec1f-92464226.png", "bdd100k_7238b169-c3aebcfe.png",
    "bdd100k_724ed3c5-a0f0504b.png", "bdd100k_73cf6448-457fc323.png", "bdd100k_746ad443-89eb8a7a.png",
    "bdd100k_750eb859-5f17f926.png", "bdd100k_7559c118-e97b44ef.png", "bdd100k_75a23bac-72c075cd.png",
    "bdd100k_76273209-14eea8f3.png", "bdd100k_772ac765-5f47e414.png", "bdd100k_77c581fe-b91cdc00.png",
    "bdd100k_77fc2c16-b82f74a7.png", "bdd100k_780039e8-9d366067.png", "bdd100k_786b3729-acc83d6f.png",
    "bdd100k_7899f781-c6b28645.png", "bdd100k_7944f396-d70990af.png", "bdd100k_79703062-77aedb97.png",
    "bdd100k_79fbe809-3c0a0517.png", "bdd100k_7a8825bf-cb837da8.png", "bdd100k_7aaf7c04-06077131.png",
    "bdd100k_7fe43deb-5d021fd3.png", "bdd100k_80af9d8e-506bc4b6.png", "bdd100k_80ce1413-656babeb.png",
    "bdd100k_80ea5511-89e4b872.png", "bdd100k_810dfc9d-0adc7e3a.png", "bdd100k_8155f34a-350b3a91.png",
    "bdd100k_816d38aa-8ab88d22.png", "bdd100k_81a98a9a-a5f57a69.png", "bdd100k_82066fbf-9c68e301.png",
    "bdd100k_82d2dd28-02174f4c.png", "bdd100k_830c6cb2-f7b30a4c.png", "bdd100k_83a0bbaf-14e491aa.png",
    "bdd100k_84f900b9-83655e13.png", "bdd100k_85ce05d6-1850506e.png", "bdd100k_87154a3d-65d8e0c4.png",
    "bdd100k_87e78191-291740fc.png", "bdd100k_88bef8d8-de477307.png", "bdd100k_88c7312d-88e0e652.png",
    "bdd100k_88d7aa59-b5d9b6b6.png", "bdd100k_88e9e61a-3ad4350e.png", "bdd100k_8ac5ec86-39e32a24.png",
    "bdd100k_8ca414ba-053096c5.png", "bdd100k_8cbbe764-20f57346.png", "bdd100k_8e6b9b3a-47a55c87.png",
    "bdd100k_906eec9d-3a4db6c5.png", "bdd100k_90952271-437d0c92.png", "bdd100k_92344fba-2059eebb.png",
    "bdd100k_930c09ad-a9b7f18f.png", "bdd100k_9340b623-7a990e3c.png", "bdd100k_939840d6-a7c62d84.png",
    "bdd100k_94ea1da6-d84cab67.png", "bdd100k_9767767c-1c92d26a.png", "bdd100k_99fda8cc-46cabc17.png",
    "bdd100k_9affc986-22d44eea.png", "bdd100k_9b0e122c-3ead3b61.png", "bdd100k_9b24e896-629e55b7.png",
    "bdd100k_9bf0d115-7c59055b.png", "bdd100k_9c38185c-fc417ba0.png", "bdd100k_9c41986f-a196bc12.png",
    "bdd100k_9fd02acf-aa351589.png", "bdd100k_9fd7efc6-a48fd2b1.png", "bdd100k_a327cb1b-04d439dc.png",
    "bdd100k_a4776c0b-2e8856a9.png", "bdd100k_a58f81f2-78ae959f.png", "bdd100k_a5ec626e-ec81a030.png",
    "bdd100k_a62b515a-b08cfa10.png", "bdd100k_a65ee719-0688ed2e.png", "bdd100k_a695eebb-1c454daa.png",
    "bdd100k_a839c26f-4c963f3a.png", "bdd100k_a8d45186-e42e9dcd.png", "bdd100k_ab8095e6-c9c6560e.png",
    "bdd100k_adb9b134-2937d690.png", "bdd100k_adfc8f5c-b1f02e04.png", "bdd100k_afd35d01-4f0f5070.png",
    "bdd100k_b10d7939-5a6b2c8e.png", "bdd100k_b11c57ab-395c5012.png", "bdd100k_fdbffb41-a07c5660.png"
]

def build_filename_map(search_dir, extensions=(".png", ".jpg", ".jpeg")):
    file_map = {}
    if not search_dir or not os.path.exists(search_dir):
        return file_map

    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(extensions):
                full_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                
                # Strip BDD suffixes automatically
                clean_stem = stem
                for suffix in [
                    "_drivable_id", "_drivable_color", "_lane_id", "_leftImg8bit",
                    "_gtFine_labellevel3Ids", "_gtFine_labellevel2Ids", 
                    "_gtFine_labellevel1Ids", "_gtFine_labelIds", "_gtFine_polygons",
                    "_gtFine", "_image", "_label"
                ]:
                    clean_stem = clean_stem.replace(suffix, "")
                
                file_map[clean_stem] = full_path
    return file_map

raw_train_dir = os.path.join("data", "raw", "bdd_lane_gt", "train")
raw_val_dir = os.path.join("data", "raw", "bdd_lane_gt", "val")
processed_val_dir = os.path.join("data", "processed", "val", "masks_lanes")

print("Scanning raw TRAIN and VAL directories...")
raw_train_map = build_filename_map(raw_train_dir)
raw_val_map = build_filename_map(raw_val_dir)

# Combine both dictionaries so we can find the file no matter where it is
combined_raw_map = {**raw_train_map, **raw_val_map}
print(f"Found {len(combined_raw_map)} total raw files.")

print(f"Hotfixing {len(wiped_files)} wiped validation lane masks...")
restored_count = 0

for target_file in wiped_files:
    # Strip everything back to the pure alphanumeric stem
    raw_stem = target_file.replace("bdd100k_", "").replace(".png", "")
    
    if raw_stem in combined_raw_map:
        actual_raw_path = combined_raw_map[raw_stem]
        raw_mask = cv2.imread(actual_raw_path, cv2.IMREAD_GRAYSCALE)
        
        if raw_mask is not None:
            lane_binary = np.where(raw_mask > 0, 255, 0).astype(np.uint8)
            lane_resized = cv2.resize(lane_binary, (640, 360), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(os.path.join(processed_val_dir, target_file), lane_resized)
            restored_count += 1

print(f"Successfully recovered {restored_count}/{len(wiped_files)} lane masks.")