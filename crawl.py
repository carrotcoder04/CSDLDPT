import os
from icrawler.builtin import BingImageCrawler

# ==========================================
# CAU HINH THONG SO
# ==========================================
MAX_IMAGES = 80          # Tai 80 anh/loai de tru hao luc loc tay
BASE_DIR = "Raw_Tree_Dataset_Test"  # Thu muc luu anh

TREE_SPECIES = {
    # "1_Pine_Tree": "pine tree full view side profile isolated",
    # "2_Weeping_Willow": "weeping willow tree full view side isolated",
    # "3_Coconut_Tree": "coconut palm tree full body side view isolated",
    # "4_Baobab_Tree": "baobab tree entire side view isolated",
    # "5_Cypress_Tree": "cypress tree full view column isolated",
    # "6_Maple_Tree": "red maple tree full view isolated",
    # "7_Birch_Tree": "white birch tree full body side view isolated",
    # "8_Cherry_Blossom": "cherry blossom tree full view bloom isolated",
    # "9_Ginkgo_Tree": "ginkgo biloba tree yellow full view isolated",
    # "10_Jacaranda_Tree": "jacaranda tree purple full view isolated",
    # "11_Oak_Tree": "mature oak tree full view side isolated",
    # "12_Banyan_Tree": "banyan tree full view roots isolated",
    # "13_Acacia_Tree": "acacia tree flat top full view isolated",
    # "14_Blue_Spruce": "blue spruce tree full view isolated",
    # "15_Eucalyptus_Tree": "eucalyptus tree full body side isolated"
    "16_Mangrove_Tree": "mangrove tree with roots full view isolated",
    "17_Dragon_Blood_Tree": "dragon blood tree socotra full view isolated",
    "18_Redwood_Tree": "giant redwood tree full view isolated",
    "19_Joshua_Tree": "joshua tree full view isolated",
    "20_Magnolia_Tree": "magnolia tree full view bloom isolated"
}

# ==========================================
# THUC THI CRAWL DU LIEU THO
# ==========================================
def crawl_raw_images_test():
    print(f"Bat dau cao du lieu test cho {len(TREE_SPECIES)} loai cay...")

    for folder_name, keyword in TREE_SPECIES.items():
        save_dir = os.path.join(BASE_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nDang tai anh cho: {folder_name} (Keyword: '{keyword}')...")

        # Dung BingImageCrawler thay GoogleImageCrawler
        # (Google thay doi HTML, GoogleImageCrawler bi loi NoneType)
        crawler = BingImageCrawler(
            parser_threads=2,
            downloader_threads=4,
            storage={'root_dir': save_dir}
        )

        try:
            crawler.crawl(keyword=keyword, max_num=MAX_IMAGES, file_idx_offset=0)
            n = len([f for f in os.listdir(save_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  => Da tai: {n} anh vao {save_dir}")
        except Exception as e:
            print(f"  [LOI] {folder_name}: {e}")

    print(f"\nHoan thanh! Kiem tra thu muc: {BASE_DIR}")


if __name__ == "__main__":
    crawl_raw_images_test()