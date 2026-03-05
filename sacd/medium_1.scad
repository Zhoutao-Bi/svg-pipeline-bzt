// ----------------------------------------------------
// Model: H-Profile Bracket / Motor Mounting Housing
// Based on SVG Slice Data Reconstruction
// Units: mm
// ----------------------------------------------------

$fn = 60; // 提高圆形/圆柱体渲染的平滑度

// 模块化构建主体
difference() {
    
    // 1. 总体外壳特征 (Square_Housing)
    // 根据切片数据边界点 (2.0 到 47.0)，计算出 45x45mm 的方形基底
    // 高度范围 (Z轴) 为 0 到 40mm
    translate([2, 2, 0]) 
        cube([45, 45, 40]); 

    // 2. 上下双空腔 (Dual_Cavity)，形成 H 型截面
    // 中间实体隔离板保留在 Z = 16mm 到 Z = 28mm 之间
    
    // 底部空腔 (Z = 0 到 16mm)
    // 预留 1mm 余量以确保完全挖穿底面
    translate([24.5, 24.5, -1]) 
        cylinder(h=17, d=38); 
        
    // 顶部空腔 (Z = 28mm 到 40mm)
    // 顶部同样留出挖空余量
    translate([24.5, 24.5, 28]) 
        cylinder(h=13, d=38); 

    // 3. 细小特征：两个水平贯穿孔 (Horizontal_Through_Holes)
    // 位于隔离板内部 (Z=22mm 处)，沿 X 轴贯穿整个壳体
    // 孔径 ~9.7mm (r=4.86)
    
    // 横向孔 1
    translate([-1, 11.3, 22]) 
        rotate([0, 90, 0]) 
        cylinder(h=55, d=9.7); 

    // 横向孔 2
    translate([-1, 37.5, 22]) 
        rotate([0, 90, 0]) 
        cylinder(h=55, d=9.7); 

    // 4. 细小特征：偏心垂直通孔 (Vertical_Offset_Hole)
    // 位于 X=33.0, Y=24.5，穿透中间的隔离板 (Z=16~28)
    // 尺寸约 13.5 x 12mm 的椭圆孔
    translate([33.0, 24.5, 15]) 
        scale([13.5/12, 1, 1]) // 拉伸形成椭圆
        cylinder(h=15, d=12); 
}