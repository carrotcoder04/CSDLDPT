import os
from icrawler.builtin import BingImageCrawler

# ==========================================
# CẤU HÌNH THÔNG SỐ
# ==========================================
MAX_IMAGES_YOUNG = 15          # Tải 15 ảnh cây non/loài
BASE_DIR = "Raw_Tree_Dataset_Test" 

# Danh sách 20 loài với từ khóa tối ưu cho CÂY NON (sapling/seedling/young)
TREE_SPECIES_YOUNG = {
    # Đợt 1
    "1_Pine_Tree": "young pine tree sapling isolated",
    "2_Weeping_Willow": "young weeping willow sapling isolated",
    "3_Coconut_Tree": "young coconut tree sapling isolated",
    "4_Baobab_Tree": "young baobab tree sapling isolated",
    "5_Cypress_Tree": "young cypress tree sapling isolated",
    
    # Đợt 2
    "6_Maple_Tree": "young maple tree sapling isolated",
    "7_Birch_Tree": "young birch tree sapling isolated",
    "8_Cherry_Blossom": "young cherry blossom sapling isolated",
    "9_Ginkgo_Tree": "young ginkgo biloba sapling isolated",
    "10_Jacaranda_Tree": "young jacaranda sapling isolated",
    
    # Bổ sung Nhóm Texture (5 loài chưa chạy đợt trước)
    "11_Oak_Tree": "young oak tree sapling isolated",
    "12_Banyan_Tree": "young banyan tree sapling isolated",
    "13_Acacia_Tree": "young acacia tree sapling isolated",
    "14_Blue_Spruce": "young blue spruce sapling isolated",
    "15_Eucalyptus_Tree": "young eucalyptus tree sapling isolated",
    
    # Đợt 3
    "16_Mangrove_Tree": "young mangrove tree sapling isolated",
    "17_Dragon_Blood_Tree": "young dragon blood tree sapling isolated",
    "18_Redwood_Tree": "young redwood tree sapling isolated",
    "19_Joshua_Tree": "young joshua tree sapling isolated",
    "20_Magnolia_Tree": "young magnolia tree sapling isolated"
}

# ==========================================
# THỰC THI CRAWL CÂY NON (KHÔNG GHI ĐÈ ẢNH CŨ)
# ==========================================
def crawl_young_trees():
    print(f"🚀 Bắt đầu cào ảnh CÂY NON cho {len(TREE_SPECIES_YOUNG)} loài...")
    
    for folder_name, keyword in TREE_SPECIES_YOUNG.items():
        save_dir = os.path.join(BASE_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Đếm số file hiện có trong thư mục để tránh ghi đè ảnh cây trưởng thành
        existing_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
        current_count = len(existing_files)
        
        print(f"\n📥 Đang tải ảnh cây non cho: {folder_name}")
        print(f"   Keyword: '{keyword}'")
        print(f"   Phát hiện {current_count} ảnh cũ. Bắt đầu lưu từ file số {current_count + 1}...")
        
        google_crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={'root_dir': save_dir}
        )
        
        # Lọc ảnh
        filters = dict(size='large', type='photo')
        
        try:
            # Truyền current_count vào file_idx_offset để viết tiếp nối tên file
            google_crawler.crawl(
                keyword=keyword, 
                filters=filters, 
                max_num=MAX_IMAGES_YOUNG, 
                file_idx_offset=current_count
            )
        except Exception as e:
            print(f"⚠️ Có lỗi khi tải {folder_name}: {e}")

    print("\n✅ Hoàn thành! Bộ dữ liệu của bạn giờ đã phong phú với cả cây non và cây trưởng thành.")

if __name__ == "__main__":
    crawl_young_trees()