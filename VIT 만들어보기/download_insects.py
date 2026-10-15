import kagglehub

path = kagglehub.dataset_download(
    "kmldas/insect-identification-from-habitus-images"
)

print("다운로드 완료:", path)