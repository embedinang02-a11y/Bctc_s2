# Xử lý chuỗi JSON trả về
    json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
    return pd.read_json(io.StringIO(json_str), typ='series')

# --- Hàm tính toán Chỉ số Tài chính (Yêu cầu 3) ---
def calculate_project_metrics(df_cashflow, initial_investment, wacc):
    """Tính toán NPV, IRR, PP, DPP."""
    
    cash_flows = df_cashflow['Dòng tiền thuần (CF)'].values
    
    # 1. NPV
    # Thêm vốn đầu tư ban đầu vào đầu dòng tiền
    full_cash_flows = np.insert(cash_flows, 0, -initial_investment) 
    npv_value = np.npv(wacc, full_cash_flows)
    
    # 2. IRR
    try:
        irr_value = np.irr(full_cash_flows)
    except ValueError:
        irr_value = np.nan # Không thể tính IRR

    # 3. PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(full_cash_flows)
    pp_year = np.where(cumulative_cf >= 0)[0]
    if pp_year.size > 0:
        pp_year = pp_year[0] # Năm mà tích lũy CF >= 0
        if pp_year == 0: 
             pp = 0 
        else:
             # Tính phân đoạn năm (năm trước - cumulative_cf) / (cf năm hoàn vốn)
             # Vốn chưa hoàn: abs(cumulative_cf[pp_year-1])
             # CF năm hoàn vốn: cash_flows[pp_year-1]
             capital_remaining = abs(cumulative_cf[pp_year-1])
             cf_of_payback_year = cash_flows[pp_year-1]
             pp = pp_year - 1 + (capital_remaining / cf_of_payback_year) if cf_of_payback_year != 0 else pp_year 
    else:
        pp = 'Không hoàn vốn'

    # 4. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discount_factors = 1 / ((1 + wacc) ** np.arange(0, len(full_cash_flows)))
    discounted_cf = full_cash_flows * discount_factors
    cumulative_dcf = np.cumsum(discounted_cf)
    
    dpp_year = np.where(cumulative_dcf >= 0)[0]
    if dpp_year.size > 0:
        dpp_year = dpp_year[0]
        if dpp_year == 0:
             dpp = 0
        else:
             capital_remaining_d = abs(cumulative_dcf[dpp_year-1])
             dcf_of_payback_year = discounted_cf[dpp_year] # Đây là DCF của năm đầu tiên mà tích lũy >= 0
             dpp = dpp_year - 1 + (capital_remaining_d / dcf_of_payback_year) if dcf_of_payback_year != 0 else dpp_year
    else:
        dpp = 'Không hoàn vốn'
        
    return npv_value, irr_value, pp, dpp

# --- Hàm gọi AI phân tích chỉ số (Yêu cầu 4) ---
def get_ai_evaluation(metrics_data, wacc_rate, api_key):
    """Gửi các chỉ số đánh giá dự án đến Gemini API và nhận phân tích."""
    
    if not api_key:
        return "Lỗi: Khóa API không được cung cấp."

    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'  

        prompt = f"""
Bạn là một chuyên gia phân tích dự án đầu tư có kinh nghiệm. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra nhận xét ngắn gọn, khách quan (khoảng 3-4 đoạn) về khả năng chấp nhận và rủi ro của dự án. 
        
        Các chỉ số cần phân tích:
        - NPV: {metrics_data['NPV']:.2f}
        - IRR: {metrics_data['IRR']:.2%}
        - WACC (Tỷ lệ chiết khấu): {wacc_rate:.2%}
        - PP (Thời gian hoàn vốn): {metrics_data['PP']} năm
        - DPP (Thời gian hoàn vốn có chiết khấu): {metrics_data['DPP']} năm
        
        Chú ý:
        1. Đánh giá tính khả thi (NPV > 0 và IRR > WACC).
        2. Nhận xét về tốc độ hoàn vốn (PP và DPP).
        3. Kết luận tổng thể về việc chấp nhận hay từ chối dự án.
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Giao diện và Luồng chính ---

# Lấy API Key
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
     st.error("⚠️ Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng AI.")

uploaded_file = st.file_uploader(
    "1. Tải file Word (.docx) chứa Phương án Kinh doanh:",
    type=['docx']
)

# Khởi tạo state để lưu trữ dữ liệu đã trích xuất
if 'extracted_data' not in st.session_state:
    st.session_state['extracted_data'] = None

# --- Chức năng 1: Lọc dữ liệu bằng AI ---
if uploaded_file is not None:
    doc_text = read_docx_file(uploaded_file)
    
    if st.button("Trích xuất Dữ liệu Tài chính bằng AI 🤖"):
        if api_key:
            with st.spinner('Đang đọc và trích xuất thông số tài chính bằng Gemini...'):
                try:
                    st.session_state['extracted_data'] = extract_financial_data(doc_text, api_key)
                    st.success("Trích xuất dữ liệu thành công!")
                except APIError:
                    st.error("Lỗi API: Không thể kết nối hoặc xác thực API Key.")
                except Exception as e:
                    st.error(f"Lỗi trích xuất: {e}")
        else:
            st.error("Vui lòng cung cấp Khóa API.")

# --- Hiển thị và Tính toán (Yêu cầu 2 & 3) ---
if st.session_state['extracted_data'] is not None:
    data = st.session_state['extracted_data']
    
    st.subheader("2. Các Thông số Dự án đã Trích xuất")
    
    # Hiển thị các thông số quan trọng (Chuyển đổi các thông số về định dạng tiền tệ/phần trăm)
    col1, col2, col3 = st.columns(3)
col1.metric("Vốn Đầu tư (C₀)", f"{data['Vốn đầu tư']:,.0f} VNĐ")
    col2.metric("Dòng đời dự án (N)", f"{data['Dòng đời dự án']:.0f} năm")
    col3.metric("WACC (k)", f"{data['WACC']:.2%}")
    col1.metric("Doanh thu Hàng năm (R)", f"{data['Doanh thu hàng năm']:,.0f} VNĐ")
    col2.metric("Chi phí HĐ Hàng năm (C)", f"{data['Chi phí hoạt động hàng năm']:,.0f} VNĐ")
    col3.metric("Thuế suất (t)", f"{data['Thuế suất']:.2%}")

    st.markdown("---")
    
    st.subheader("3. Bảng Dòng tiền (Cash Flow)")
    
    # Giả định: Khấu hao = Vốn đầu tư / Dòng đời dự án (phương pháp đường thẳng)
    # Giả định: Giá trị thanh lý (Salvage Value) = 0
    initial_investment = data['Vốn đầu tư']
    project_life = int(data['Dòng đời dự án'])
    annual_revenue = data['Doanh thu hàng năm']
    annual_cost = data['Chi phí hoạt động hàng năm']
    wacc = data['WACC']
    tax_rate = data['Thuế suất']
    
    try:
        depreciation = initial_investment / project_life if project_life > 0 else 0
    except ZeroDivisionError:
        st.error("Dòng đời dự án phải lớn hơn 0 để tính khấu hao.")
        depreciation = 0

    # Bảng dòng tiền
    years = np.arange(1, project_life + 1)
    
    EBT = annual_revenue - annual_cost - depreciation
    Tax = EBT * tax_rate if EBT > 0 else 0
    EAT = EBT - Tax
    # Dòng tiền thuần = Lợi nhuận sau thuế + Khấu hao
    CF = EAT + depreciation
    
    cashflow_data = {
        'Năm': years,
        'Doanh thu (R)': [annual_revenue] * project_life,
        'Chi phí HĐ (C)': [annual_cost] * project_life,
        'Khấu hao (D)': [depreciation] * project_life,
        'Lợi nhuận trước thuế (EBT)': [EBT] * project_life,
        'Thuế (Tax)': [Tax] * project_life,
        'Lợi nhuận sau thuế (EAT)': [EAT] * project_life,
        'Dòng tiền thuần (CF)': [CF] * project_life
    }
    
    df_cashflow = pd.DataFrame(cashflow_data)
    
    st.dataframe(
        df_cashflow.style.format({
            col: '{:,.0f}' for col in df_cashflow.columns if col not in ['Năm']
        }), 
        use_container_width=True
    )

    st.markdown("---")
    
    st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
    
    try:
        npv, irr, pp, dpp = calculate_project_metrics(df_cashflow, initial_investment, wacc)
        
        metrics_data = {
            'NPV': npv,
            'IRR': irr,
            'PP': pp,
            'DPP': dpp
        }
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NPV (Giá trị hiện tại thuần)", f"{npv:,.0f} VNĐ", delta=("Dự án có lời" if npv > 0 else "Dự án lỗ"))
        col2.metric("IRR (Tỷ suất sinh lời nội tại)", f"{irr:.2%}" if not np.isnan(irr) else "N/A")
col3.metric("PP (Thời gian hoàn vốn)", f"{pp:.2f} năm" if isinstance(pp, float) else pp)
        col4.metric("DPP (Hoàn vốn có chiết khấu)", f"{dpp:.2f} năm" if isinstance(dpp, float) else dpp)

        # --- Chức năng 5: Yêu cầu AI Phân tích ---
        st.markdown("---")
        st.subheader("5. Phân tích Hiệu quả Dự án (AI)")
        
        if st.button("Yêu cầu AI Phân tích Chỉ số 🧠"):
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_evaluation(metrics_data, wacc, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                 st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng kiểm tra cấu hình Secrets.")

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi tính toán chỉ số: {e}. Vui lòng kiểm tra các thông số đầu vào.")

else:
    st.info("Vui lòng tải lên file Word và nhấn nút 'Trích xuất Dữ liệu Tài chính bằng AI' để bắt đầu.")
